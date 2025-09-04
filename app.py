# Disable ChromaDB telemetry BEFORE any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import streamlit as st
import asyncio
import uuid
import time
import logging
from dotenv import load_dotenv
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('tripfix_app.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that sends messages to Streamlit session state"""
    def emit(self, record):
        try:
            # Only add agent-related logs to frontend
            if any(keyword in record.getMessage() for keyword in [
                "🧠", "🌍", "⚖️", "📊", "🔍", "✅", "❌", "🔄", "💬", "📝", "⏱️", "📄"
            ]):
                # This will be called from agent modules, so we need to check if we're in a Streamlit context
                import streamlit as st
                if hasattr(st, 'session_state') and 'agent_activity_log' in st.session_state:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_entry = {
                        "timestamp": timestamp,
                        "level": record.levelname,
                        "message": record.getMessage()
                    }
                    st.session_state.agent_activity_log.append(log_entry)
                    # Keep only last 20 entries
                    if len(st.session_state.agent_activity_log) > 20:
                        st.session_state.agent_activity_log = st.session_state.agent_activity_log[-20:]
        except:
            # Ignore errors if not in Streamlit context
            pass

# Add the custom handler to agent loggers
streamlit_handler = StreamlitLogHandler()
agents_logger = logging.getLogger('agents')
agents_logger.addHandler(streamlit_handler)
agents_logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

# Import our components
from agents.intake_agent import IntakeAgent
from utils.database import IntakeDatabase  
from utils.vector_store import VectorStore
from utils.performance_tracker import get_performance_tracker, track_performance, track_session
import base64

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
    
    .risk-level-low {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 5px 0;
    }
    
    .risk-level-medium {
        background: linear-gradient(90deg, #ffc107, #fd7e14);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 5px 0;
    }
    
    .risk-level-high {
        background: linear-gradient(90deg, #fd7e14, #dc3545);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 5px 0;
    }
    
    .risk-level-critical {
        background: linear-gradient(90deg, #dc3545, #6f42c1);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 5px 0;
        animation: pulse 1s ease-in-out infinite alternate;
    }
    
    .risk-factor {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 10px;
        margin: 5px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .risk-factor-high {
        border-left-color: #dc3545;
    }
    
    .risk-factor-medium {
        border-left-color: #ffc107;
    }
    
    .risk-factor-low {
        border-left-color: #28a745;
    }
    
    .pattern-detected {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        padding: 8px 12px;
        border-radius: 15px;
        margin: 3px;
        display: inline-block;
        font-size: 0.9em;
    }
    
    .handoff-priority {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
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

if "agent_activity_log" not in st.session_state:
    st.session_state.agent_activity_log = []

def add_agent_log(message: str, level: str = "INFO"):
    """Add an entry to the agent activity log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message
    }
    st.session_state.agent_activity_log.append(log_entry)
    # Keep only last 20 entries
    if len(st.session_state.agent_activity_log) > 20:
        st.session_state.agent_activity_log = st.session_state.agent_activity_log[-20:]

def display_risk_assessment(session_data):
    """Display the Advanced Confidence Engine risk assessment"""
    if not session_data or not session_data.get('risk_assessment'):
        return
    
    try:
        risk_assessment = json.loads(session_data['risk_assessment'])
    except (json.JSONDecodeError, TypeError):
        return
    
    st.subheader("🎯 Risk Assessment")
    
    # Overall confidence and risk level
    confidence = risk_assessment.get('overall_confidence', 0)
    risk_level = risk_assessment.get('risk_level', 'unknown')
    
    # Risk level display
    risk_level_class = f"risk-level-{risk_level}"
    risk_level_emoji = {
        'low': '🟢',
        'medium': '🟡', 
        'high': '🟠',
        'critical': '🔴'
    }.get(risk_level, '⚪')
    
    st.markdown(f"""
    <div class="{risk_level_class}">
        {risk_level_emoji} {risk_level.upper()} RISK ({confidence:.1%} confidence)
    </div>
    """, unsafe_allow_html=True)
    
    # Handoff information
    if risk_assessment.get('handoff_required', False):
        priority = risk_assessment.get('handoff_priority', 'Unknown')
        st.markdown(f"""
        <div class="handoff-priority">
            ⚠️ <strong>Human Review Required</strong><br>
            Priority: {priority}
        </div>
        """, unsafe_allow_html=True)
    
    # Risk factors breakdown
    st.subheader("📊 Risk Factors")
    risk_factors = risk_assessment.get('risk_factors', [])
    
    for factor in risk_factors:
        name = factor.get('name', 'Unknown')
        score = factor.get('score', 0)
        weight = factor.get('weight', 0)
        reasoning = factor.get('reasoning', 'No details available')
        
        # Determine factor class based on score
        if score >= 0.8:
            factor_class = "risk-factor-low"
            emoji = "🟢"
        elif score >= 0.6:
            factor_class = "risk-factor-medium"
            emoji = "🟡"
        else:
            factor_class = "risk-factor-high"
            emoji = "🔴"
        
        st.markdown(f"""
        <div class="risk-factor {factor_class}">
            <strong>{emoji} {name}</strong> ({weight:.0%} weight)<br>
            Score: {score:.2f}<br>
            <small>{reasoning}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Patterns detected
    patterns = risk_assessment.get('patterns_detected', [])
    if patterns:
        st.subheader("🔍 Patterns Detected")
        pattern_html = ""
        for pattern in patterns:
            pattern_html += f'<span class="pattern-detected">🔍 {pattern}</span>'
        st.markdown(pattern_html, unsafe_allow_html=True)
    
    # Contextual factors
    contextual_factors = risk_assessment.get('contextual_factors', [])
    if contextual_factors:
        st.subheader("💬 Contextual Factors")
        for factor in contextual_factors:
            st.markdown(f"• {factor}")

@st.cache_resource
def initialize_system():
    """Initialize the AI system components"""
    try:
        logger.info("🚀 Starting TripFix AI system initialization...")
        
        # Initialize performance tracker
        logger.info("📊 Initializing performance tracker...")
        performance_tracker = get_performance_tracker()
        logger.info("✅ Performance tracker initialized")
        
        # Initialize database
        logger.info("🗄️ Initializing database...")
        database = IntakeDatabase()
        logger.info("✅ Database initialized")
        
        # Initialize vector store with improved chunking
        logger.info("🔍 Initializing vector store and loading regulations...")
        vector_store = VectorStore(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vector_store.initialize_from_pdfs()
        logger.info("✅ Vector store initialized with regulation documents")
        
        # Initialize main agent
        logger.info("🤖 Initializing IntakeAgent with specialized sub-agents...")
        agent = IntakeAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            database=database,
            vector_store=vector_store
        )
        logger.info("✅ IntakeAgent initialized with JurisdictionAgent, EligibilityAgent, and Advanced Confidence Engine")
        
        logger.info("🎉 TripFix AI system fully initialized and ready!")
        return agent, database, vector_store, performance_tracker
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        st.error(f"Failed to initialize system: {e}")
        return None, None, None, None

# Main app
def main():
    logger.info("🌐 TripFix frontend application starting...")
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ TripFix Flight Delay Compensation</h1>
        <p>Get the compensation you deserve for flight delays and cancellations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    with st.spinner("🔧 Initializing AI system..."):
        logger.info("🔄 Frontend requesting AI system initialization...")
        add_agent_log("🚀 Starting TripFix AI system initialization...", "INFO")
        agent, database, vector_store, performance_tracker = initialize_system()
        add_agent_log("✅ TripFix AI system fully initialized and ready!", "INFO")
    
    if not agent:
        st.error("Failed to initialize the system. Please check your OpenAI API key and try again.")
        return
    
    # Sidebar with session info and risk assessment
    with st.sidebar:
        st.header("Session Information")
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        if database:
            session_data = database.get_session(st.session_state.session_id)
            if session_data:
                st.write(f"**Status:** {session_data.get('status', 'New')}")
                st.write(f"**Created:** {session_data.get('created_at', 'Unknown')}")
                
                # Display risk assessment if available
                display_risk_assessment(session_data)
        
        if st.button("🔄 Start New Session"):
            # Track session end
            if performance_tracker:
                performance_tracker.track_session_end(st.session_state.session_id)
            
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.agent_status = ""
            
            # Track new session start
            if performance_tracker:
                performance_tracker.track_session_start(st.session_state.session_id)
            
            st.rerun()
        
        # Performance metrics in sidebar
        if performance_tracker:
            st.header("📊 Performance")
            current_metrics = performance_tracker.get_current_metrics()
            st.metric("Response Time", f"{current_metrics['avg_response_time']:.1f}s")
            st.metric("Active Sessions", current_metrics['active_sessions'])
            st.metric("Success Rate", f"{(1 - current_metrics['error_rate']):.1%}")
    
    # Chat interface
    st.header("💬 Chat with TripFix Assistant")
    
    # Chat container
    chat_container = st.container()
    
    # Display messages
    with chat_container:
        # Display conversation history
        for message in st.session_state.messages:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            step = message.get("step", "")
            
            with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
                if step and role == "assistant":
                    # Enhanced status display for Advanced Confidence Engine
                    if step == "assessing_eligibility":
                        st.caption("🧠 Advanced Confidence Engine: Analyzing risk factors...")
                    elif step == "handoff_to_human":
                        st.caption("⚠️ Risk Assessment: Human review required")
                    elif step == "completed":
                        st.caption("✅ Risk Assessment: Auto-processed with high confidence")
                    else:
                        st.caption(f"🔄 Agent Status: {step.replace('_', ' ').title()}")
                st.markdown(content)
    
    # Agent status indicator with detailed logging
    if st.session_state.agent_status:
        st.markdown(f"""
        <div class="agent-status">
            🤖 Agent Status: {st.session_state.agent_status}
        </div>
        """, unsafe_allow_html=True)
    
    # Real-time agent activity log
    if st.session_state.get("agent_activity_log"):
        with st.expander("🔍 Agent Activity Log", expanded=False):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"Showing {len(st.session_state.agent_activity_log)} recent activities")
            with col2:
                if st.button("Clear Log", key="clear_agent_log"):
                    st.session_state.agent_activity_log = []
                    st.rerun()
            
            # Show log entries in reverse order (newest first)
            for log_entry in reversed(st.session_state.agent_activity_log[-15:]):  # Show last 15 entries
                timestamp = log_entry.get("timestamp", "")
                level = log_entry.get("level", "INFO")
                message = log_entry.get("message", "")
                
                # Color code based on level
                if level == "ERROR":
                    st.error(f"❌ {timestamp} - {message}")
                elif level == "WARNING":
                    st.warning(f"⚠️ {timestamp} - {message}")
                else:
                    st.info(f"ℹ️ {timestamp} - {message}")
    
    # Processing indicator with live status
    if st.session_state.processing:
        st.markdown('<div class="progress-indicator"></div>', unsafe_allow_html=True)
        
        # Show current agent activity
        if st.session_state.get("agent_activity_log"):
            latest_activity = st.session_state.agent_activity_log[-1] if st.session_state.agent_activity_log else None
            if latest_activity:
                st.caption(f"🤖 Currently: {latest_activity.get('message', 'Processing...')}")
    
    # Initialize conversation if this is a new session
    if not st.session_state.messages:
        st.session_state.processing = True
        st.session_state.agent_status = "Initializing conversation..."
        
        # Track session start
        if performance_tracker:
            performance_tracker.track_session_start(st.session_state.session_id)
        
        try:
            # Show processing status
            with st.chat_message("assistant", avatar="🤖"):
                status_placeholder = st.empty()
                status_placeholder.caption("🔄 Agent Status: Starting conversation...")
                
                # Process with agent to get initial greeting
                async def process_async():
                    return await agent.process_message(st.session_state.session_id, "start")
                
                # Run async function
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
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
            st.error(f"Error initializing conversation: {e}")
        
        finally:
            st.session_state.processing = False
            st.session_state.agent_status = ""
            st.rerun()

    # File upload section - only show after delay reason is collected
    progress = agent.get_intake_progress(st.session_state.session_id)
    should_offer_upload = (
        progress.get("delay_reason_collected", False) and
        not progress.get("supporting_files_offered", False)
    )
    
    if should_offer_upload:
        st.subheader("📎 Supporting Documents")
        
        # Ask user if they want to upload documents
        st.info("💡 Do you have any supporting documents (boarding passes, delay notifications, etc.) that could help with your case?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, I have documents to upload", disabled=st.session_state.processing):
                st.session_state.show_upload = True
                st.rerun()
        with col2:
            if st.button("No, I don't have documents", disabled=st.session_state.processing):
                # Mark that supporting files were offered and user declined
                agent.database.update_intake_progress(st.session_state.session_id, supporting_files_offered=True)
                st.session_state.show_upload = False
                
                # Send a message to the agent to continue the workflow
                user_message = {
                    "role": "user",
                    "content": "No, I don't have any supporting documents",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(user_message)
                
                # Process the message with the agent
                st.session_state.processing = True
                st.session_state.agent_status = "Processing your response..."
                
                try:
                    # Show processing status
                    with st.chat_message("user", avatar="👤"):
                        st.markdown("No, I don't have any supporting documents")
                    
                    with st.chat_message("assistant", avatar="🤖"):
                        status_placeholder = st.empty()
                        status_placeholder.caption("🔄 Agent Status: Processing your response...")
                        
                        # Process with agent
                        async def process_async():
                            return await agent.process_message(st.session_state.session_id, "No, I don't have any supporting documents")
                        
                        # Run async function
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        # Update status to show Advanced Confidence Engine is working
                        status_placeholder.caption("🧠 Advanced Confidence Engine: Processing multi-factor risk assessment...")
                        
                        # Log analysis start
                        logger.info(f"🧠 Starting Advanced Confidence Engine analysis for session {st.session_state.session_id[:8]}...")
                        logger.info(f"📝 User message: No, I don't have any supporting documents")
                        
                        # Add to frontend log
                        add_agent_log("🧠 Starting Advanced Confidence Engine analysis...", "INFO")
                        add_agent_log("📝 User declined supporting documents, continuing to analysis...", "INFO")
                        
                        # Track performance
                        start_time = time.time()
                        result = loop.run_until_complete(process_async())
                        processing_time = time.time() - start_time
                        
                        logger.info(f"⏱️ Analysis completed in {processing_time:.2f} seconds")
                        add_agent_log(f"⏱️ Analysis completed in {processing_time:.2f} seconds", "INFO")
                        
                        # Track the performance metric
                        if performance_tracker:
                            performance_tracker.track_metric(
                                component="intake_agent",
                                operation="process_message",
                                duration=processing_time,
                                success=True,
                                metadata={"session_id": st.session_state.session_id, "message_length": len("No, I don't have any supporting documents")},
                                session_id=st.session_state.session_id
                            )
                        
                        # Get the latest assistant message
                        assistant_messages = [msg for msg in result["messages"] if msg["role"] == "assistant"]
                        if assistant_messages:
                            latest_message = assistant_messages[-1]
                            
                            # Update status
                            step = latest_message.get("step", "")
                            if step:
                                status_placeholder.caption(f"🔄 Agent Status: {step.replace('_', ' ').title()}")
                                logger.info(f"🔄 Agent workflow step: {step}")
                            
                            # Log the response
                            logger.info(f"💬 Agent response generated: {latest_message['content'][:100]}{'...' if len(latest_message['content']) > 100 else ''}")
                            
                            # Show response
                            st.markdown(latest_message["content"])
                            
                            # Add to session state
                            st.session_state.messages.append(latest_message)
                
                except Exception as e:
                    st.error(f"Error processing message: {e}")
                    logger.error(f"Error processing 'no documents' response: {e}")
                    
                    # Track error
                    if performance_tracker:
                        performance_tracker.track_metric(
                            component="intake_agent",
                            operation="process_message",
                            duration=time.time() - start_time,
                            success=False,
                            metadata={"error": str(e), "session_id": st.session_state.session_id},
                            session_id=st.session_state.session_id
                        )
                
                finally:
                    st.session_state.processing = False
                    st.session_state.agent_status = ""
                    st.rerun()
        
        # Show upload interface if user chose to upload
        if st.session_state.get("show_upload", False):
            uploaded_file = st.file_uploader(
                "Upload supporting documents",
                type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'doc', 'docx'],
                help="Supported formats: PDF, images (PNG, JPG, JPEG), text files, Word documents",
                disabled=st.session_state.processing
            )
            
            if uploaded_file is not None:
                # Process the uploaded file
                with st.spinner("Processing uploaded file..."):
                    file_content = uploaded_file.read()
                    result = agent.process_file_upload(
                        st.session_state.session_id,
                        file_content,
                        uploaded_file.name
                    )
                    
                    if result["success"]:
                        st.success(result["message"])
                        
                        # Show extracted information if available
                        if result.get("extracted_flight_info"):
                            flight_info = result["extracted_flight_info"]
                            if any(flight_info.values()):
                                st.write("**Information extracted from document:**")
                                if flight_info["flight_numbers"]:
                                    st.write(f"• Flight numbers: {', '.join(flight_info['flight_numbers'])}")
                                if flight_info["airlines"]:
                                    st.write(f"• Airlines: {', '.join(flight_info['airlines'])}")
                                if flight_info["dates"]:
                                    st.write(f"• Dates: {', '.join(flight_info['dates'])}")
                                if flight_info["airports"]:
                                    st.write(f"• Airports: {', '.join(flight_info['airports'])}")
                                if flight_info["delay_info"]:
                                    st.write(f"• Delay information: {', '.join([f'{info[0]} {info[1]}' for info in flight_info['delay_info']])}")
                        
                        # Show text preview
                        if result.get("extracted_text_preview"):
                            with st.expander("View extracted text"):
                                st.text(result["extracted_text_preview"])
                    else:
                        st.error(f"Failed to process file: {result.get('error', 'Unknown error')}")
    
    # Show uploaded files
    supporting_files = agent.get_supporting_files(st.session_state.session_id)
    if supporting_files:
        st.write("**Uploaded Documents:**")
        for file_info in supporting_files:
            st.write(f"📄 {file_info['filename']} ({file_info['file_type']}, {file_info['file_size']} bytes)")
    
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
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Update status to show Advanced Confidence Engine is working
                status_placeholder.caption("🧠 Advanced Confidence Engine: Processing multi-factor risk assessment...")
                
                # Log analysis start
                logger.info(f"🧠 Starting Advanced Confidence Engine analysis for session {st.session_state.session_id[:8]}...")
                logger.info(f"📝 User message: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                
                # Add to frontend log
                add_agent_log("🧠 Starting Advanced Confidence Engine analysis...", "INFO")
                add_agent_log(f"📝 Processing user message: {prompt[:50]}{'...' if len(prompt) > 50 else ''}", "INFO")
                
                # Track performance
                start_time = time.time()
                result = loop.run_until_complete(process_async())
                processing_time = time.time() - start_time
                
                logger.info(f"⏱️ Analysis completed in {processing_time:.2f} seconds")
                add_agent_log(f"⏱️ Analysis completed in {processing_time:.2f} seconds", "INFO")
                
                # Track the performance metric
                if performance_tracker:
                    performance_tracker.track_metric(
                        component="intake_agent",
                        operation="process_message",
                        duration=processing_time,
                        success=True,
                        metadata={"session_id": st.session_state.session_id, "message_length": len(prompt)},
                        session_id=st.session_state.session_id
                    )
                
                # Get the latest assistant message
                assistant_messages = [msg for msg in result["messages"] if msg["role"] == "assistant"]
                if assistant_messages:
                    latest_message = assistant_messages[-1]
                    
                    # Update status
                    step = latest_message.get("step", "")
                    if step:
                        status_placeholder.caption(f"🔄 Agent Status: {step.replace('_', ' ').title()}")
                        logger.info(f"🔄 Agent workflow step: {step}")
                        add_agent_log(f"🔄 Agent workflow step: {step.replace('_', ' ').title()}", "INFO")
                    
                    # Log the response
                    logger.info(f"💬 Agent response generated: {latest_message['content'][:100]}{'...' if len(latest_message['content']) > 100 else ''}")
                    add_agent_log(f"💬 Agent response generated", "INFO")
                    
                    # Show response
                    st.markdown(latest_message["content"])
                    
                    # Add to session state
                    st.session_state.messages.append(latest_message)
        
        except Exception as e:
            st.error(f"Error processing message: {e}")
            
            # Track error
            if performance_tracker:
                performance_tracker.track_metric(
                    component="intake_agent",
                    operation="process_message",
                    duration=time.time() - start_time,
                    success=False,
                    metadata={"error": str(e), "session_id": st.session_state.session_id},
                    session_id=st.session_state.session_id
                )
        
        finally:
            st.session_state.processing = False
            st.session_state.agent_status = ""
            st.rerun()

if __name__ == "__main__":
    logger.info("🎯 TripFix application entry point - starting main function")
    main()