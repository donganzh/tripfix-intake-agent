import streamlit as st
import asyncio
import uuid
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Import our components
from agents.intake_agent import IntakeAgent
from utils.database import IntakeDatabase  
from utils.vector_store import VectorStore

# Page configuration
st.set_page_config(
    page_title="TripFix - Flight Delay Compensation",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern chat interface
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .user-message {
        background: #007bff;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        margin-left: 20%;
        text-align: right;
    }
    
    .assistant-message {
        background: white;
        border: 1px solid #e9ecef;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        margin-right: 20%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .agent-status {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 10px;
        border-radius: 8px;
        margin: 8px 0;
        font-style: italic;
        color: #856404;
    }
    
    .progress-indicator {
        background: linear-gradient(90deg, #00b4db, #0083b0);
        height: 4px;
        border-radius: 2px;
        margin: 10px 0;
        animation: pulse 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        from { opacity: 0.6; }
        to { opacity: 1.0; }
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_status" not in st.session_state:
    st.session_state.agent_status = ""

if "processing" not in st.session_state:
    st.session_state.processing = False

@st.cache_resource
def initialize_system():
    """Initialize the AI system components"""
    try:
        # Initialize database
        database = IntakeDatabase()
        
        # Initialize vector store
        vector_store = VectorStore(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vector_store.initialize_from_pdfs()
        
        # Initialize main agent
        agent = IntakeAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            database=database,
            vector_store=vector_store
        )
        
        return agent, database, vector_store
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None, None

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ TripFix Flight Delay Compensation</h1>
        <p>Get the compensation you deserve for flight delays and cancellations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    with st.spinner("🔧 Initializing AI system..."):
        agent, database, vector_store = initialize_system()
    
    if not agent:
        st.error("Failed to initialize the system. Please check your OpenAI API key and try again.")
        return
    
    # Sidebar with session info
    with st.sidebar:
        st.header("Session Information")
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        if database:
            session_data = database.get_session(st.session_state.session_id)
            if session_data:
                st.write(f"**Status:** {session_data.get('status', 'New')}")
                st.write(f"**Created:** {session_data.get('created_at', 'Unknown')}")
        
        if st.button("🔄 Start New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.agent_status = ""
            st.experimental_rerun()
    
    # Chat interface
    st.header("💬 Chat with TripFix Assistant")
    
    # Chat container
    chat_container = st.container()
    
    # Display messages
    with chat_container:
        if not st.session_state.messages:
            # Show initial greeting
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
                👋 **Welcome to TripFix!** I'm here to help you understand your rights and potentially get compensation for your flight delay.

I understand how frustrating flight delays can be, especially when they disrupt important plans. Let's work together to see if you're eligible for compensation under air passenger rights laws.

To get started, I'll need to gather some information about your flight. Can you please tell me your **flight number** and the **date** of your delayed flight?
                """)
        
        # Display conversation history
        for message in st.session_state.messages:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            step = message.get("step", "")
            
            with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
                if step and role == "assistant":
                    st.caption(f"🔄 Agent Status: {step.replace('_', ' ').title()}")
                st.markdown(content)
    
    # Agent status indicator
    if st.session_state.agent_status:
        st.markdown(f"""
        <div class="agent-status">
            🤖 Agent Status: {st.session_state.agent_status}
        </div>
        """, unsafe_allow_html=True)
    
    # Processing indicator
    if st.session_state.processing:
        st.markdown('<div class="progress-indicator"></div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message here...", disabled=st.session_state.processing):
        # Add user message to session state
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Show user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Process message
        st.session_state.processing = True
        st.session_state.agent_status = "Processing your request..."
        
        try:
            # Show processing status
            with st.chat_message("assistant", avatar="🤖"):
                status_placeholder = st.empty()
                status_placeholder.caption("🔄 Agent Status: Analyzing your message...")
                
                # Process with agent
                async def process_async():
                    return await agent.process_message(st.session_state.session_id, prompt)
                
                # Run async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(process_async())
                
                # Get the latest assistant message
                assistant_messages = [msg for msg in result["messages"] if msg["role"] == "assistant"]
                if assistant_messages:
                    latest_message = assistant_messages[-1]
                    
                    # Update status
                    step = latest_message.get("step", "")
                    if step:
                        status_placeholder.caption(f"🔄 Agent Status: {step.replace('_', ' ').title()}")
                    
                    # Show response
                    st.markdown(latest_message["content"])
                    
                    # Add to session state
                    st.session_state.messages.append(latest_message)
        
        except Exception as e:
            st.error(f"Error processing message: {e}")
        
        finally:
            st.session_state.processing = False
            st.session_state.agent_status = ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()