from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, TypedDict, Optional
import json
import uuid
import logging
from datetime import datetime

# Configure logging for agents
logger = logging.getLogger(__name__)

from agents.jurisdiction_agent import JurisdictionAgent
from agents.eligibility_agent import EligibilityAgent
from agents.confidence_scorer import ConfidenceScorer
from agents.advanced_confidence_engine import AdvancedConfidenceEngine, RiskLevel
from utils.database import IntakeDatabase
from utils.vector_store import VectorStore
from utils.file_processor import get_file_processor

class IntakeState(TypedDict):
    session_id: str
    messages: List[Dict[str, str]]
    flight_data: Dict[str, Any]
    current_step: str
    jurisdiction: Optional[str]
    jurisdiction_confidence: Optional[float]
    jurisdiction_reasoning: Optional[str]
    eligibility_result: Optional[Dict[str, Any]]
    eligibility_confidence: Optional[float]
    needs_handoff: bool
    handoff_reason: Optional[str]
    handoff_priority: Optional[str]
    risk_level: Optional[str]
    risk_assessment: Optional[Dict[str, Any]]
    completed: bool
    next_question: Optional[str]
    user_name: Optional[str]
    user_mood: Optional[str]
    # Feedback loop states
    awaiting_feedback: bool
    feedback_iteration: int
    user_satisfied: Optional[bool]
    additional_info_provided: bool
    escalation_required: bool

class IntakeAgent:
    def __init__(self, openai_api_key: str, database: IntakeDatabase, vector_store: VectorStore):
        logger.info("🤖 Initializing IntakeAgent...")
        
        self.openai_api_key = openai_api_key
        self.database = database
        self.vector_store = vector_store
        self.file_processor = get_file_processor()
        
        logger.info("🧠 Setting up main LLM (GPT-4o-mini)...")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=openai_api_key,
            temperature=0.3
        )
        
        # Initialize specialized agents
        logger.info("🌍 Initializing JurisdictionAgent...")
        self.jurisdiction_agent = JurisdictionAgent(openai_api_key, vector_store)
        logger.info("✅ JurisdictionAgent initialized")
        
        logger.info("⚖️ Initializing EligibilityAgent...")
        self.eligibility_agent = EligibilityAgent(openai_api_key, vector_store)
        logger.info("✅ EligibilityAgent initialized")
        
        logger.info("📊 Initializing ConfidenceScorer...")
        self.confidence_scorer = ConfidenceScorer()  # Keep for backward compatibility
        logger.info("✅ ConfidenceScorer initialized")
        
        logger.info("🧠 Initializing Advanced Confidence Engine...")
        self.advanced_confidence_engine = AdvancedConfidenceEngine()
        logger.info("✅ Advanced Confidence Engine initialized")
        
        # Required fields for intake
        self.required_fields = [
            'user_name', 'user_mood', 'flight_number', 'flight_date', 'airline', 'origin', 
            'destination', 'connecting_airports', 'delay_length', 'delay_reason'
        ]
        
        logger.info("🔄 Creating LangGraph workflow...")
        self.graph = self.create_workflow()
        logger.info("✅ IntakeAgent fully initialized with all sub-agents and workflow")
    
    def create_workflow(self):
        """Create the LangGraph workflow"""
        workflow = StateGraph(IntakeState)
        
        # Add nodes
        workflow.add_node("greet", self.greet_user)
        workflow.add_node("collect_info", self.collect_flight_info)
        workflow.add_node("validate_data", self.validate_flight_data)
        workflow.add_node("collect_documents", self.collect_supporting_documents)
        workflow.add_node("determine_jurisdiction", self.determine_jurisdiction)
        workflow.add_node("assess_eligibility", self.assess_eligibility)
        workflow.add_node("score_confidence", self.score_confidence)
        workflow.add_node("handoff_human", self.handoff_to_human)
        workflow.add_node("complete_intake", self.complete_intake)
        workflow.add_node("handle_off_topic", self.handle_off_topic)
        workflow.add_node("handle_feedback", self.handle_user_feedback)
        workflow.add_node("reopen_analysis", self.reopen_analysis)
        workflow.add_node("provide_guidance", self.provide_guidance)
        
        # Set entry point
        workflow.set_entry_point("greet")
        
        # Add conditional edges - simplified to avoid recursion
        workflow.add_conditional_edges(
            "collect_info",
            self.should_validate_data,
            {
                "validate": "validate_data",
                "continue_collecting": END,  # End here to wait for user input
                "off_topic": "handle_off_topic"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_data",
            self.data_validation_next,
            {
                "documents": "collect_documents",
                "collect_more": END  # End here to wait for user input
            }
        )
        
        workflow.add_conditional_edges(
            "collect_documents",
            self.document_collection_next,
            {
                "jurisdiction": "determine_jurisdiction",
                "continue_documents": END  # End here to wait for user input
            }
        )
        
        workflow.add_conditional_edges(
            "score_confidence",
            self.confidence_decision,
            {
                "handoff": "handoff_human",
                "complete": "complete_intake"
            }
        )
        
        # Add conditional edges for feedback loop
        workflow.add_conditional_edges(
            "complete_intake",
            self.should_await_feedback,
            {
                "await_feedback": "handle_feedback",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "handle_feedback",
            self.feedback_decision,
            {
                "reopen": "reopen_analysis",
                "guidance": "provide_guidance",
                "escalate": "handoff_human",
                "end": END
            }
        )
        
        workflow.add_edge("reopen_analysis", "collect_info")
        workflow.add_edge("provide_guidance", END)
        
        # Simple edges
        workflow.add_edge("greet", "collect_info")
        workflow.add_edge("determine_jurisdiction", "assess_eligibility")
        workflow.add_edge("assess_eligibility", "score_confidence")
        workflow.add_edge("handoff_human", END)
        workflow.add_edge("handle_off_topic", "collect_info")
        
        return workflow.compile()
    
    def greet_user(self, state: IntakeState) -> IntakeState:
        """Initial greeting and setup"""
        
        # Generate dynamic greeting using LLM with much more variety
        import random
        
        greeting_styles = [
            "warm_welcoming",
            "empathetic_understanding", 
            "professional_helpful",
            "friendly_supportive",
            "caring_advisor"
        ]
        
        selected_greeting_style = random.choice(greeting_styles)
        
        greeting_style_instructions = {
            "warm_welcoming": "Be warm and welcoming, like greeting a friend. Use phrases like 'Welcome!' or 'So glad you reached out'.",
            "empathetic_understanding": "Show deep empathy and understanding. Use phrases like 'I know how frustrating this must be' or 'Flight delays are never easy'.",
            "professional_helpful": "Be professional but approachable. Use phrases like 'I'm here to help' or 'Let's get this sorted for you'.",
            "friendly_supportive": "Be friendly and supportive. Use phrases like 'Don't worry, we'll help you' or 'You're in good hands'.",
            "caring_advisor": "Be caring and advisory. Use phrases like 'I'm here to guide you' or 'Let's work through this together'."
        }
        
        greeting_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are Agent S, a customer service agent for TripFix, a company that helps passengers get compensation for flight delays.

GREETING STYLE: {selected_greeting_style}
STYLE INSTRUCTIONS: {greeting_style_instructions[selected_greeting_style]}

Generate a warm, welcoming greeting that:
1. Introduces yourself as Agent S and TripFix
2. Shows empathy for flight delay frustrations
3. Explains that you'll help them understand their rights and potential compensation
4. Asks for their name and how they're doing today
5. Then mentions you'll need their flight details

CRITICAL REQUIREMENTS:
1. Make it completely unique and natural
2. Use the selected greeting style authentically
3. Sound like a real human, not a chatbot
4. Be conversational and engaging
5. Show genuine personality and interest
6. Avoid generic phrases like "I understand" or "I'm sorry"
7. Make it feel personal and warm

Be conversational, professional, and understanding. Keep it concise but warm. Always refer to yourself as Agent S."""),
            ("human", "Generate a unique, welcoming greeting for a new customer.")
        ])
        
        try:
            # Use a more creative LLM instance for varied greetings
            from langchain_openai import ChatOpenAI
            creative_llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=self.openai_api_key,
                temperature=0.9  # Even higher temperature for greetings
            )
            
            chain = greeting_prompt | creative_llm
            response = chain.invoke({})
            greeting = response.content
        except Exception as e:
            print(f"Error generating greeting: {e}")
            # Fallback greeting if LLM fails
            greeting = """👋 Welcome to TripFix! I'm Agent S, and I'm here to help you understand your rights and potentially get compensation for your flight delay.

I understand how frustrating flight delays can be, especially when they disrupt important plans. Let's work together to see if you're eligible for compensation under air passenger rights laws.

To get started, could you please tell me your name and how you're doing today? Then I'll need to gather some information about your flight."""
        
        state["messages"].append({
            "role": "assistant",
            "content": greeting,
            "timestamp": datetime.now().isoformat()
        })
        
        state["current_step"] = "collecting_user_info"
        return state
    
    def collect_flight_info(self, state: IntakeState) -> IntakeState:
        """Collect user info and flight information with empathetic, dynamic questions"""
        
        # Check if we have a user message to process
        last_user_message = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        # If we have a user message, try to extract user info and flight info
        if last_user_message and last_user_message != "start":
            # Extract user info first if we're in the user info collection phase
            if state["current_step"] == "collecting_user_info":
                user_info = self.extract_user_info(last_user_message)
                if user_info.get("user_name"):
                    state["user_name"] = user_info["user_name"]
                if user_info.get("user_mood"):
                    state["user_mood"] = user_info["user_mood"]
            else:
                # Extract flight info
                previous_flight_data = state["flight_data"].copy()
                state["flight_data"] = self.extract_flight_info(last_user_message, state["flight_data"])
                
                # Track progress for delay_reason specifically
                if (not previous_flight_data.get("delay_reason") and 
                    state["flight_data"].get("delay_reason")):
                    # Update progress to mark delay_reason as collected
                    self.database.update_intake_progress(state["session_id"], delay_reason_collected=True)
        
        # Check if we've already responded to the last user message
        # Only proceed if we haven't responded yet OR if we just processed new information
        if last_user_message and last_user_message != "start":
            # Check if we've already responded to this user message
            last_user_message_index = -1
            for i, msg in enumerate(reversed(state["messages"])):
                if msg["role"] == "user" and msg["content"] == last_user_message:
                    last_user_message_index = len(state["messages"]) - 1 - i
                    break
            
            if last_user_message_index >= 0:
                has_responded = False
                for i in range(last_user_message_index + 1, len(state["messages"])):
                    if state["messages"][i]["role"] == "assistant":
                        has_responded = True
                        break
                
                # If we've already responded to this exact user message, don't ask again
                if has_responded:
                    return state
        
        # Determine what information we still need
        missing_fields = []
        
        # Check user info first
        if not state.get("user_name"):
            missing_fields.append("user_name")
        if not state.get("user_mood"):
            missing_fields.append("user_mood")
        
        # If we have user info, check flight data
        if state.get("user_name") and state.get("user_mood"):
            for field in ['flight_number', 'flight_date', 'airline', 'origin', 'destination', 'connecting_airports', 'delay_length', 'delay_reason']:
                if field not in state["flight_data"] or not state["flight_data"][field]:
                    missing_fields.append(field)
        
        if not missing_fields:
            # All required information collected, move to validation
            state["current_step"] = "validated"
            
            # Generate a simple message that we have all the info and will ask about documents
            user_name = state.get("user_name", "there")
            completion_message = f"Perfect! I have all the information I need about your flight delay, {user_name}. Now I just need to ask about any supporting documents you might have."
            
            state["messages"].append({
                "role": "assistant",
                "content": completion_message,
                "timestamp": datetime.now().isoformat(),
                "step": "info_collection_complete"
            })
            
            return state
        
        # Generate contextual question based on what's missing
        next_field = missing_fields[0]
        previous_data = state["flight_data"]
        user_name = state.get("user_name", "")
        user_mood = state.get("user_mood", "")
        
        # Special handling for user info collection
        if next_field in ["user_name", "user_mood"]:
            if not user_name and not user_mood:
                # First time asking for both name and mood
                question = "Could you please tell me your name and how you're doing today?"
                state["current_step"] = "collecting_user_info"
            elif not user_name:
                question = "I'd love to know your name so I can address you properly!"
                state["current_step"] = "collecting_user_info"
            elif not user_mood:
                question = f"Nice to meet you, {user_name}! How are you doing today?"
                state["current_step"] = "collecting_user_info"
        else:
            # Transition to flight info collection
            if state["current_step"] == "collecting_user_info":
                state["current_step"] = "collecting_flight_info"
                
                # Generate varied transition message
                import random
                
                transition_styles = [
                    "enthusiastic_helper",
                    "empathetic_guide",
                    "professional_assistant",
                    "caring_supporter"
                ]
                
                selected_style = random.choice(transition_styles)
                
                style_instructions = {
                    "enthusiastic_helper": "Be enthusiastic and helpful. Use phrases like 'Great!' or 'Perfect!' or 'Excellent!'",
                    "empathetic_guide": "Be empathetic and guiding. Use phrases like 'I understand' or 'Let's work through this together'",
                    "professional_assistant": "Be professional and efficient. Use phrases like 'Now let's proceed' or 'Let's get started'",
                    "caring_supporter": "Be caring and supportive. Use phrases like 'I'm here to help' or 'Don't worry, we'll get this sorted'"
                }
                
                transition_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are Agent S, a customer service agent for TripFix.

TRANSITION STYLE: {selected_style}
STYLE INSTRUCTIONS: {style_instructions[selected_style]}

Generate a natural transition message to move from user info collection to flight info collection.

CONTEXT: We just collected the user's name ({user_name}) and mood ({user_mood}). Now we need to start collecting their flight information.

CRITICAL REQUIREMENTS:
1. Make it completely unique and natural
2. Use the selected transition style authentically
3. Sound like a real human, not a chatbot
4. Be conversational and engaging
5. Acknowledge their name and mood
6. Transition smoothly to asking for flight details
7. Ask for flight number and date
8. Make it feel personal and warm

Generate a single, natural transition message."""),
                    ("human", "Generate a unique transition message.")
                ])
                
                try:
                    # Use a more creative LLM instance for varied responses
                    from langchain_openai import ChatOpenAI
                    creative_llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        openai_api_key=self.openai_api_key,
                        temperature=0.8
                    )
                    
                    chain = transition_prompt | creative_llm
                    response = chain.invoke({})
                    question = response.content
                except Exception as e:
                    print(f"Error generating transition message: {e}")
                    question = f"Thank you, {user_name}! I'm glad to hear you're {user_mood}. Now, let's get started with your flight delay case. Could you please tell me your flight number and the date of your delayed flight?"
            else:
                # Regular flight info questions with much more variety and personality
                import random
                
                # Define different conversation styles and personalities
                conversation_styles = [
                    "empathetic_helper",
                    "professional_advisor", 
                    "friendly_neighbor",
                    "understanding_friend",
                    "efficient_specialist",
                    "caring_supporter"
                ]
                
                selected_style = random.choice(conversation_styles)
                
                style_instructions = {
                    "empathetic_helper": "Be deeply empathetic and understanding. Show genuine care for their situation. Use phrases like 'I can only imagine how frustrating this must be' or 'That sounds really difficult'.",
                    "professional_advisor": "Be professional but warm. Show expertise and confidence. Use phrases like 'Let me help you with that' or 'I'll make sure we get this sorted'.",
                    "friendly_neighbor": "Be casual and friendly, like talking to a neighbor. Use phrases like 'Oh no, that's terrible!' or 'I'm so sorry that happened to you'.",
                    "understanding_friend": "Be like a supportive friend who really gets it. Use phrases like 'I totally understand' or 'That's so unfair'.",
                    "efficient_specialist": "Be efficient and focused on helping. Use phrases like 'Let's get this resolved quickly' or 'I'll make this as smooth as possible'.",
                    "caring_supporter": "Be caring and supportive. Use phrases like 'I'm here to help you through this' or 'We'll get you the compensation you deserve'."
                }
                
                # Add contextual variety based on what we already know
                contextual_elements = []
                if previous_data.get('flight_number'):
                    contextual_elements.append(f"their flight {previous_data['flight_number']}")
                if previous_data.get('airline'):
                    contextual_elements.append(f"with {previous_data['airline']}")
                if previous_data.get('origin'):
                    contextual_elements.append(f"from {previous_data['origin']}")
                
                context_string = " and ".join(contextual_elements) if contextual_elements else "their flight"
                
                question_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are Agent S, a customer service agent for TripFix helping passengers with flight delay compensation.

CONVERSATION STYLE: {selected_style}
STYLE INSTRUCTIONS: {style_instructions[selected_style]}

CONTEXT: We're helping {user_name} get compensation for their flight delay. They mentioned they're {user_mood}.
CURRENT INFO: We know about {context_string}
NEXT FIELD NEEDED: {next_field}

Field descriptions:
- flight_number: The specific flight number (e.g., AC123, WF456)
- flight_date: Date of the delayed flight
- airline: Name of the airline
- origin: Departure airport/city
- destination: Arrival airport/city
- connecting_airports: Any connecting airports (yes/no and details if yes)
- delay_length: How many hours the flight was delayed
- delay_reason: What reason the airline gave for the delay

CRITICAL REQUIREMENTS:
1. Make each response completely unique and natural
2. Vary your language, tone, and approach significantly
3. Use the selected conversation style authentically
4. Reference what we already know about their situation
5. Sound like a real human having a conversation, not a chatbot
6. Avoid repetitive phrases like "I completely understand" or "I'm sorry to hear"
7. Be conversational and engaging
8. Show personality and genuine interest

Generate a single, natural question that sounds completely human and unique. Make it feel like a real conversation."""),
                    ("human", "Generate a unique, natural question to ask for {next_field}")
                ])
                
                try:
                    # Use a more creative LLM instance for varied responses
                    from langchain_openai import ChatOpenAI
                    creative_llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        openai_api_key=self.openai_api_key,
                        temperature=0.8  # Higher temperature for more creativity
                    )
                    
                    chain = question_prompt | creative_llm
                    response = chain.invoke({
                        "next_field": next_field,
                        "collected_data": json.dumps(previous_data, indent=2),
                        "user_name": user_name,
                        "user_mood": user_mood
                    })
                    question = response.content
                except Exception as e:
                    print(f"Error generating question: {e}")
                    # Fallback to simple question
                    question = f"Could you please provide your {next_field.replace('_', ' ')}?"
        
        state["messages"].append({
            "role": "assistant", 
            "content": question,
            "timestamp": datetime.now().isoformat(),
            "step": "collecting_info",
            "collecting_field": next_field
        })
        
        return state
    
    def should_validate_data(self, state: IntakeState) -> str:
        """Decide next step based on collected data"""
        
        # Check if user message is off-topic
        last_user_message = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                last_user_message = msg["content"].lower()
                break
        
        off_topic_indicators = [
            "weather", "restaurant", "hotel", "car rental", "vacation", "hawaii", "beach",
            "unrelated", "something else", "different topic"
        ]
        
        if any(indicator in last_user_message for indicator in off_topic_indicators):
            if not any(flight_word in last_user_message for flight_word in ["flight", "delay", "airline", "airport"]):
                return "off_topic"
        
        # Check if we have all required data
        missing_fields = []
        for field in self.required_fields:
            if field not in state["flight_data"] or not state["flight_data"][field]:
                missing_fields.append(field)
        
        # If we have all required data, validate
        if not missing_fields:
            return "validate"
        
        # If we're missing data, we need to continue collecting
        # But we'll end the workflow here to wait for user input
        return "continue_collecting"
    
    def validate_flight_data(self, state: IntakeState) -> IntakeState:
        """Validate collected flight data"""
        validation_issues = []
        
        # Validate flight date
        flight_date = state["flight_data"].get("flight_date")
        if flight_date:
            # Basic date validation logic here
            pass
        
        # Validate delay length
        try:
            delay = float(state["flight_data"].get("delay_length", 0))
            if delay < 0:
                validation_issues.append("Delay length cannot be negative")
            state["flight_data"]["delay_length"] = delay
        except:
            validation_issues.append("Invalid delay length format")
        
        if validation_issues:
            state["current_step"] = "collect_info"
            
            # Generate validation error message using LLM
            validation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an empathetic customer service agent for TripFix. 

Generate a helpful, understanding message to inform the customer about validation issues with their flight information. Be gentle and supportive.

Validation issues: {validation_issues}

Generate a message that:
1. Acknowledges the issues without being critical
2. Offers to help clarify the details
3. Maintains a supportive, helpful tone"""),
                ("human", "Generate a validation error message.")
            ])
            
            try:
                chain = validation_prompt | self.llm
                response = chain.invoke({"validation_issues": "; ".join(validation_issues)})
                validation_message = response.content
            except Exception as e:
                print(f"Error generating validation message: {e}")
                validation_message = f"I notice some issues with the information provided: {'; '.join(validation_issues)}. Let's clarify these details."
            
            state["messages"].append({
                "role": "assistant",
                "content": validation_message,
                "timestamp": datetime.now().isoformat()
            })
        else:
            state["current_step"] = "validated"
        
        return state
    
    def data_validation_next(self, state: IntakeState) -> str:
        """Determine next step after validation"""
        # If validation passed, proceed to document collection
        if state["current_step"] == "validated":
            return "documents"
        else:
            # Need to collect more data - end workflow to wait for user input
            return "collect_more"
    
    def collect_supporting_documents(self, state: IntakeState) -> IntakeState:
        """Collect supporting documents from user"""
        logger.info(f"📄 Collecting supporting documents for session {state['session_id'][:8]}...")
        
        # Check if we have a user message to process
        last_user_message = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        logger.info(f"📝 Last user message: {last_user_message[:50]}{'...' if len(last_user_message) > 50 else ''}")
        
        # If we have a user message, process their document response
        if last_user_message and last_user_message != "start":
            # Extract document preference from user message
            document_preference = self.extract_document_preference(last_user_message)
            logger.info(f"📄 Extracted document preference: {document_preference}")
            if document_preference:
                state["flight_data"]["document_preference"] = document_preference
                logger.info(f"✅ Set document preference in flight data: {document_preference}")
        
        # Check if we've already responded to the last user message
        if last_user_message and last_user_message != "start":
            # Check if we've already responded to this user message
            last_user_message_index = -1
            for i, msg in enumerate(reversed(state["messages"])):
                if msg["role"] == "user" and msg["content"] == last_user_message:
                    last_user_message_index = len(state["messages"]) - 1 - i
                    break
            
            if last_user_message_index >= 0:
                has_responded = False
                for i in range(last_user_message_index + 1, len(state["messages"])):
                    if state["messages"][i]["role"] == "assistant":
                        has_responded = True
                        break
                
                # If we've already responded to this exact user message, don't ask again
                if has_responded:
                    return state
        
        # Check if we need to ask about documents
        logger.info(f"📄 Current document preference: {state['flight_data'].get('document_preference')}")
        if not state["flight_data"].get("document_preference"):
            logger.info("📄 No document preference set, asking user about documents...")
            # Generate document collection question
            import random
            
            document_styles = [
                "helpful_advisor",
                "efficient_collector",
                "caring_supporter",
                "professional_assistant"
            ]
            
            selected_style = random.choice(document_styles)
            
            style_instructions = {
                "helpful_advisor": "Be helpful and advisory. Use phrases like 'This could really help your case' or 'Documents can strengthen your claim'.",
                "efficient_collector": "Be efficient and direct. Use phrases like 'Let's gather everything we need' or 'This will help us process faster'.",
                "caring_supporter": "Be caring and supportive. Use phrases like 'I want to make sure we have everything' or 'This will help us help you better'.",
                "professional_assistant": "Be professional and thorough. Use phrases like 'Supporting documents can be valuable' or 'This will help with your case'."
            }
            
            user_name = state.get("user_name", "there")
            
            document_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are Agent S, a customer service agent for TripFix.

DOCUMENT STYLE: {selected_style}
STYLE INSTRUCTIONS: {style_instructions[selected_style]}

Generate a natural question asking if the customer has any supporting documents for their flight delay case.

CONTEXT: We're helping {user_name} with their flight delay compensation claim.

CRITICAL REQUIREMENTS:
1. Make it completely unique and natural
2. Use the selected document style authentically
3. Sound like a real human, not a chatbot
4. Be conversational and engaging
5. Explain why documents are helpful
6. Ask if they have documents like boarding passes, delay notifications, etc.
7. Make it feel personal and helpful

Generate a single, natural question about supporting documents."""),
                ("human", "Generate a unique question about supporting documents.")
            ])
            
            try:
                # Use a more creative LLM instance for varied responses
                from langchain_openai import ChatOpenAI
                creative_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=self.openai_api_key,
                    temperature=0.8
                )
                
                chain = document_prompt | creative_llm
                response = chain.invoke({})
                question = response.content
            except Exception as e:
                print(f"Error generating document question: {e}")
                question = f"Hi {user_name}! Do you have any supporting documents for your flight delay case, such as boarding passes, delay notifications, or other relevant paperwork? These can really help strengthen your compensation claim."
            
            state["messages"].append({
                "role": "assistant", 
                "content": question,
                "timestamp": datetime.now().isoformat(),
                "step": "collecting_documents"
            })
            
            state["current_step"] = "collecting_documents"
            logger.info("📄 Set current step to 'collecting_documents'")
        else:
            # Documents preference collected, move to jurisdiction
            state["current_step"] = "documents_collected"
            logger.info("📄 Document preference collected, moving to 'documents_collected' step")
            
            # Add analysis message here where it should appear
            analysis_message = {
                "role": "assistant",
                "content": "🔍 Analyzing your case... Please wait while I process your information.",
                "timestamp": datetime.now().isoformat(),
                "step": "analyzing_progress"
            }
            state["messages"].append(analysis_message)
        
        logger.info(f"📄 Final state step: {state['current_step']}")
        return state
    
    def document_collection_next(self, state: IntakeState) -> str:
        """Determine next step after document collection"""
        # If documents preference collected, proceed to jurisdiction
        if state["current_step"] == "documents_collected":
            return "jurisdiction"
        else:
            # Need to continue collecting documents - end workflow to wait for user input
            return "continue_documents"
    
    def extract_document_preference(self, user_message: str) -> str:
        """Extract document preference from user message"""
        message_lower = user_message.lower()
        
        # Check for negative responses first (more specific)
        if any(phrase in message_lower for phrase in ["no, i don't", "don't have", "do not have", "no documents", "no supporting", "none", "nothing"]):
            return "no"
        # Check for positive responses
        elif any(word in message_lower for word in ["yes", "have", "do have", "sure", "okay", "ok"]):
            return "yes"
        
        return None
    
    def determine_jurisdiction(self, state: IntakeState) -> IntakeState:
        """Determine applicable jurisdiction"""
        logger.info(f"🌍 Starting jurisdiction determination for session {state['session_id'][:8]}...")
        
        # Generate jurisdiction analysis message using LLM
        jurisdiction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Agent S, a helpful customer service agent for TripFix. 

Generate a professional, reassuring message to inform the customer that you're analyzing their flight details to determine which air passenger rights laws apply to their situation.

Be:
1. Professional and confident
2. Reassuring that you're working on their case
3. Brief but informative about what you're doing
4. You can refer to yourself as Agent S if it feels natural"""),
            ("human", "Generate a message about analyzing jurisdiction.")
        ])
        
        try:
            chain = jurisdiction_prompt | self.llm
            response = chain.invoke({})
            jurisdiction_message = response.content
        except Exception as e:
            print(f"Error generating jurisdiction message: {e}")
            jurisdiction_message = "🔍 Let me analyze your flight details to determine which air passenger rights laws apply to your situation..."
        
        state["messages"].append({
            "role": "assistant",
            "content": jurisdiction_message,
            "timestamp": datetime.now().isoformat(),
            "step": "analyzing_jurisdiction"
        })
        
        # Add progress indicator
        progress_message = {
            "role": "assistant",
            "content": "🌍 Determining applicable regulations...",
            "timestamp": datetime.now().isoformat(),
            "step": "jurisdiction_progress"
        }
        state["messages"].append(progress_message)
        
        logger.info("🔍 Calling JurisdictionAgent to analyze flight data...")
        jurisdiction, reasoning, citations = self.jurisdiction_agent.determine_jurisdiction(
            state["flight_data"]
        )
        logger.info(f"✅ Jurisdiction determined: {jurisdiction}")
        
        # Score confidence
        logger.info("📊 Scoring jurisdiction confidence...")
        confidence, confidence_explanation = self.confidence_scorer.score_jurisdiction_confidence(
            state["flight_data"], jurisdiction, reasoning
        )
        logger.info(f"📊 Jurisdiction confidence: {confidence:.2f}")
        
        state["jurisdiction"] = jurisdiction
        state["jurisdiction_confidence"] = confidence
        state["jurisdiction_reasoning"] = reasoning
        state["current_step"] = "jurisdiction_determined"
        
        # Store in database
        self.database.update_session(
            state["session_id"],
            jurisdiction=jurisdiction,
            jurisdiction_confidence=confidence,
            flight_data=state["flight_data"]
        )
        
        return state
    
    def assess_eligibility(self, state: IntakeState) -> IntakeState:
        """Assess eligibility for compensation"""
        logger.info(f"⚖️ Starting eligibility assessment for session {state['session_id'][:8]}...")
        
        # Generate eligibility analysis message using LLM
        eligibility_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Agent S, a helpful customer service agent for TripFix. 

Generate a professional, reassuring message to inform the customer that you're analyzing their eligibility for compensation under the applicable regulations.

Jurisdiction: {jurisdiction}

Be:
1. Professional and confident
2. Reassuring that you're working on their case
3. Brief but informative about what you're doing
4. You can refer to yourself as Agent S if it feels natural"""),
            ("human", "Generate a message about analyzing eligibility.")
        ])
        
        try:
            chain = eligibility_prompt | self.llm
            response = chain.invoke({"jurisdiction": state['jurisdiction']})
            eligibility_message = response.content
        except Exception as e:
            print(f"Error generating eligibility message: {e}")
            eligibility_message = f"⚖️ Analyzing your eligibility under {state['jurisdiction']} regulations..."
        
        state["messages"].append({
            "role": "assistant", 
            "content": eligibility_message,
            "timestamp": datetime.now().isoformat(),
            "step": "analyzing_eligibility"
        })
        
        # Add progress indicator
        progress_message = {
            "role": "assistant",
            "content": "⚖️ Assessing eligibility for compensation...",
            "timestamp": datetime.now().isoformat(),
            "step": "eligibility_progress"
        }
        state["messages"].append(progress_message)
        
        logger.info("🔍 Calling EligibilityAgent to analyze compensation eligibility...")
        eligible, compensation, reasoning, citations = self.eligibility_agent.assess_eligibility(
            state["flight_data"], 
            state["jurisdiction"]
        )
        logger.info(f"✅ Eligibility assessment complete: Eligible={eligible}, Compensation=${compensation}")
        
        state["eligibility_result"] = {
            "eligible": eligible,
            "compensation_amount": compensation,
            "reasoning": reasoning,
            "legal_citations": citations
        }
        state["current_step"] = "eligibility_assessed"
        
        # Continue to confidence scoring and completion
        result = self.score_confidence(state)
        if result["needs_handoff"]:
            result = self.handoff_to_human(result)
        else:
            result = self.complete_intake(result)
        
        return result
    
    def score_confidence(self, state: IntakeState) -> IntakeState:
        """Score confidence using Advanced Confidence Engine and determine if handoff needed"""
        logger.info(f"🧠 Starting Advanced Confidence Engine risk assessment for session {state['session_id'][:8]}...")
        
        # Get conversation history for contextual analysis
        conversation_history = self.database.get_conversation_history(state["session_id"])
        logger.info(f"📚 Retrieved {len(conversation_history)} conversation messages for contextual analysis")
        
        # Perform comprehensive risk assessment
        logger.info("🔍 Calling Advanced Confidence Engine for multi-factor risk assessment...")
        risk_assessment = self.advanced_confidence_engine.assess_risk(
            flight_data=state["flight_data"],
            jurisdiction_result=state["jurisdiction"],
            jurisdiction_reasoning=state["jurisdiction_reasoning"],
            eligibility_result=state["eligibility_result"],
            conversation_history=conversation_history
        )
        logger.info(f"✅ Risk assessment complete: Risk Level={risk_assessment.risk_level.value}, Confidence={risk_assessment.overall_confidence:.2f}, Handoff Required={risk_assessment.handoff_required}")
        
        # Add progress indicator for confidence assessment
        confidence_progress_message = {
            "role": "assistant",
            "content": "🧠 Performing risk assessment and confidence analysis...",
            "timestamp": datetime.now().isoformat(),
            "step": "confidence_progress"
        }
        state["messages"].append(confidence_progress_message)
        
        # Update state with risk assessment results
        state["eligibility_confidence"] = risk_assessment.overall_confidence
        state["needs_handoff"] = risk_assessment.handoff_required
        state["handoff_reason"] = risk_assessment.reasoning
        state["handoff_priority"] = risk_assessment.handoff_priority
        state["risk_level"] = risk_assessment.risk_level.value
        state["risk_assessment"] = {
            "overall_confidence": risk_assessment.overall_confidence,
            "risk_level": risk_assessment.risk_level.value,
            "risk_factors": [
                {
                    "name": factor.name,
                    "weight": factor.weight,
                    "score": factor.score,
                    "reasoning": factor.reasoning,
                    "multiplier": factor.multiplier
                }
                for factor in risk_assessment.risk_factors
            ],
            "patterns_detected": risk_assessment.patterns_detected,
            "contextual_factors": risk_assessment.contextual_factors,
            "handoff_required": risk_assessment.handoff_required,
            "handoff_priority": risk_assessment.handoff_priority
        }
        
        # Update database with comprehensive risk assessment
        self.database.update_session(
            state["session_id"],
            eligibility_result=json.dumps(state["eligibility_result"]),
            eligibility_confidence=risk_assessment.overall_confidence,
            handoff_reason=risk_assessment.reasoning if risk_assessment.handoff_required else None,
            risk_assessment=json.dumps(state["risk_assessment"])
        )
        
        return state
    
    def confidence_decision(self, state: IntakeState) -> str:
        """Decide whether to handoff or complete"""
        return "handoff" if state["needs_handoff"] else "complete"
    
    def handoff_to_human(self, state: IntakeState) -> IntakeState:
        """Hand off to human specialist"""
        
        # Generate handoff message using LLM
        handoff_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Agent S, a helpful customer service agent for TripFix. 

Generate a professional, reassuring message to inform the customer that their case is being handed off to a human specialist for review.

Handoff reason: {handoff_reason}

The message should:
1. Explain that a human specialist will review their case
2. Mention the reason for handoff in a positive way
3. Set expectations (24 hours, email notification)
4. Thank them and express commitment to helping them
5. Be professional but warm and reassuring
6. You can refer to yourself as Agent S if it feels natural"""),
            ("human", "Generate a handoff message.")
        ])
        
        try:
            chain = handoff_prompt | self.llm
            response = chain.invoke({"handoff_reason": state['handoff_reason']})
            handoff_message = response.content
        except Exception as e:
            print(f"Error generating handoff message: {e}")
            handoff_message = f"""🔄 I've analyzed your case, but due to its complexity, I'm connecting you with one of our human specialists for the most accurate assessment.

**Why a specialist is reviewing your case:** {state['handoff_reason']}

A TripFix expert will review your flight details within 24 hours and provide you with a comprehensive analysis of your compensation eligibility. You'll receive an email with their findings and next steps.

Thank you for choosing TripFix - we're committed to getting you the compensation you deserve! 🛫✈️"""
        
        state["messages"].append({
            "role": "assistant",
            "content": handoff_message,
            "timestamp": datetime.now().isoformat(),
            "step": "handoff_complete"
        })
        
        # Mark for human review in database
        self.database.update_session(
            state["session_id"],
            status="human_review_required",
            completed=False
        )
        
        state["completed"] = True
        return state
    
    def handle_follow_up_question(self, state: IntakeState) -> IntakeState:
        """Handle follow-up questions after guidance has been provided"""
        logger.info(f"🔄 Handling follow-up question for session {state['session_id'][:8]}...")
        
        # Get the last user message
        last_user_message = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            logger.warning("No user message found for follow-up question")
            return state
        
        logger.info(f"📝 Processing follow-up question: {last_user_message[:50]}...")
        
        # Analyze the follow-up question
        follow_up_analysis = self.analyze_follow_up_question(last_user_message, state)
        logger.info(f"📊 Follow-up analysis: {follow_up_analysis}")
        
        # Generate contextual response based on analysis
        response = self.generate_follow_up_response(follow_up_analysis, state)
        logger.info(f"📝 Generated response: {response[:100]}...")
        
        # Add response to messages
        response_message = {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
            "step": "follow_up_response"
        }
        state["messages"].append(response_message)
        
        # Update state based on response type
        if follow_up_analysis.get("should_end_chat", False):
            state["current_step"] = "chat_ended"
            state["completed"] = True
            logger.info("✅ Chat ended based on follow-up analysis")
        else:
            state["current_step"] = "guidance_provided"  # Keep in guidance state for more questions
            logger.info("✅ Kept in guidance_provided state for more questions")
        
        return state
    
    def analyze_follow_up_question(self, user_message: str, state: IntakeState) -> Dict[str, Any]:
        """Analyze follow-up questions to determine appropriate response"""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing a follow-up question from a user who has already received guidance about their flight delay compensation case.

Context:
- User has completed the intake process
- User has received eligibility assessment and guidance
- This is a follow-up question after initial guidance

User's follow-up question: {user_message}

Analyze the question and determine:
1. Question type (clarification, escalation, additional info, goodbye, etc.)
2. Whether the user wants to escalate to human agent
3. Whether the user is asking for more specific information
4. Whether the user is ready to end the conversation
5. The appropriate response approach

Respond with JSON:
{{
    "question_type": "clarification|escalation|additional_info|goodbye|other",
    "wants_human_agent": true/false,
    "needs_specific_info": true/false,
    "ready_to_end": true/false,
    "should_end_chat": true/false,
    "response_approach": "clarify|escalate|provide_info|goodbye|continue_guidance",
    "key_points": ["point1", "point2", "point3"]
}}"""),
            ("human", "Analyze this follow-up question.")
        ])
        
        try:
            chain = analysis_prompt | self.llm
            response = chain.invoke({"user_message": user_message})
            
            # Parse JSON response
            analysis = json.loads(response.content)
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing follow-up question: {e}")
            # Fallback analysis
            return {
                "question_type": "other",
                "wants_human_agent": "agent" in user_message.lower() or "human" in user_message.lower(),
                "needs_specific_info": False,
                "ready_to_end": False,
                "should_end_chat": False,
                "response_approach": "continue_guidance",
                "key_points": []
            }
    
    def generate_follow_up_response(self, analysis: Dict[str, Any], state: IntakeState) -> str:
        """Generate contextual response to follow-up questions"""
        
        user_name = state.get("user_name", "there")
        eligibility_result = state.get("eligibility_result", {})
        is_eligible = eligibility_result.get("eligible", False)
        compensation = eligibility_result.get("compensation_amount", 0)
        
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Agent S from TripFix, providing follow-up assistance to a user who has already received their eligibility assessment and initial guidance.

User Context:
- Name: {user_name}
- Eligible: {is_eligible}
- Compensation: ${compensation}
- Question Type: {question_type}
- Response Approach: {response_approach}

Analysis: {analysis}

Generate a helpful, contextual response that:
1. Addresses the user's specific follow-up question
2. Provides relevant information based on their eligibility status
3. Guides them toward appropriate next steps
4. If they want human agent, explain the escalation process
5. If they're ready to end, provide a warm goodbye
6. Keep the tone professional, helpful, and empathetic

Be concise but comprehensive. If this seems like a natural ending point, guide them to conclude the conversation."""),
            ("human", "Generate a follow-up response.")
        ])
        
        try:
            chain = response_prompt | self.llm
            response = chain.invoke({
                "user_name": user_name,
                "is_eligible": is_eligible,
                "compensation": compensation,
                "question_type": analysis.get("question_type", "other"),
                "response_approach": analysis.get("response_approach", "continue_guidance"),
                "analysis": json.dumps(analysis, indent=2)
            })
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating follow-up response: {e}")
            # Fallback response
            if analysis.get("wants_human_agent", False):
                return f"Thank you for your question, {user_name}. I understand you'd like to speak with a human agent. I'll escalate your case to our specialist team who will contact you within 24 hours. Is there anything else I can help you with today?"
            else:
                return f"Thank you for your question, {user_name}. I'm here to help with any additional questions you might have about your case. Is there anything specific you'd like to know more about?"
    
    def complete_intake(self, state: IntakeState) -> IntakeState:
        """Complete the automated intake process"""
        eligibility = state["eligibility_result"]
        
        # Generate completion message using LLM
        completion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Agent S, a helpful customer service agent for TripFix. 

Generate a professional completion message for the customer based on their eligibility assessment.

Customer Name: {user_name}
Eligibility: {eligible}
Compensation Amount: {compensation_amount}
Jurisdiction: {jurisdiction}
Reasoning: {reasoning}

If eligible:
- Congratulate them on being eligible using their name
- Mention the compensation amount
- Explain the legal basis
- Outline next steps
- Express commitment to helping them
- Include their name in the email body context

If not eligible:
- Be empathetic and understanding, using their name
- Explain why they don't qualify
- Suggest alternative options
- Offer manual review as an option

Be professional, empathetic, and helpful in both cases. You can refer to yourself as Agent S if it feels natural. Always address them by name when appropriate."""),
            ("human", "Generate a completion message.")
        ])
        
        try:
            # Use a more creative LLM instance for varied completion messages
            from langchain_openai import ChatOpenAI
            creative_llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=self.openai_api_key,
                temperature=0.7  # Higher temperature for more creative responses
            )
            
            chain = completion_prompt | creative_llm
            response = chain.invoke({
                "user_name": state.get("user_name", "there"),
                "eligible": eligibility["eligible"],
                "compensation_amount": eligibility.get('compensation_amount', 0),
                "jurisdiction": state['jurisdiction'],
                "reasoning": eligibility['reasoning']
            })
            completion_message = response.content
        except Exception as e:
            print(f"Error generating completion message: {e}")
            # Fallback messages
            user_name = state.get("user_name", "there")
            if eligibility["eligible"]:
                completion_message = f"""✅ **Great news, {user_name}! You appear to be eligible for compensation!**

**Compensation Amount:** ${eligibility['compensation_amount']:.2f} CAD

**Legal Basis:** Your case falls under {state['jurisdiction']} regulations. {eligibility['reasoning']}

**Next Steps:**
1. We'll prepare your claim documentation
2. Our legal team will contact the airline on your behalf
3. You'll receive updates as we progress your case

Welcome to TripFix, {user_name}! We're committed to getting you the compensation you deserve. 🛫💰"""
            else:
                completion_message = f"""Unfortunately, {user_name}, based on our analysis, your flight delay doesn't appear to qualify for compensation under applicable {state['jurisdiction']} regulations.

**Reason:** {eligibility['reasoning']}

However, you may still have other options:
- Travel insurance claims
- Airline goodwill gestures
- Vouchers or credits

Our team can still review your case manually if you'd like, {user_name}. Sometimes there are nuances that our automated system might miss."""
        
        state["messages"].append({
            "role": "assistant",
            "content": completion_message,
            "timestamp": datetime.now().isoformat(),
            "step": "intake_complete"
        })
        
        # Set up feedback loop
        state["awaiting_feedback"] = True
        state["feedback_iteration"] = 0
        state["user_satisfied"] = None
        state["additional_info_provided"] = False
        state["escalation_required"] = False
        
        # Update database
        self.database.update_session(
            state["session_id"],
            status="awaiting_feedback",
            completed=False,
            flight_data=json.dumps(state["flight_data"]),
            jurisdiction=state.get("jurisdiction"),
            jurisdiction_confidence=state.get("jurisdiction_confidence"),
            eligibility_result=json.dumps(state.get("eligibility_result", {})),
            eligibility_confidence=state.get("eligibility_confidence")
        )
        
        state["completed"] = False  # Keep as False to allow feedback
        return state
    
    def handle_off_topic(self, state: IntakeState) -> IntakeState:
        """Handle off-topic questions"""
        
        # Get the last user message to understand what they're asking about
        last_user_message = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        # Generate contextual off-topic redirect message using completely different approaches
        import random
        
        # Define completely different response styles to force variation
        response_styles = [
            "casual_friendly",
            "professional_formal", 
            "empathetic_understanding",
            "direct_helpful",
            "conversational_natural"
        ]
        
        selected_style = random.choice(response_styles)
        
        style_templates = {
            "casual_friendly": f"""Respond in a casual, friendly way. Acknowledge their question about "{last_user_message}" but explain you're here to help with flight delays. Be like a helpful friend who's redirecting them to what you can actually help with.""",
            
            "professional_formal": f"""Respond professionally and formally. Acknowledge their inquiry about "{last_user_message}" and explain that your area of expertise is flight delay compensation. Be polite but clear about your specialization.""",
            
            "empathetic_understanding": f"""Show empathy and understanding. Acknowledge that their question about "{last_user_message}" is important to them, but explain that you're specifically trained to help with flight delay compensation. Be warm and understanding.""",
            
            "direct_helpful": f"""Be direct and helpful. Acknowledge their question about "{last_user_message}" and directly explain that you can't help with that, but you can help with flight delay compensation. Be straightforward and helpful.""",
            
            "conversational_natural": f"""Respond like you're having a natural conversation. Acknowledge their question about "{last_user_message}" and naturally steer the conversation toward flight delay compensation. Be conversational and natural."""
        }
        
        redirect_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are Agent S, a helpful customer service agent for TripFix, specializing in flight delay compensation claims.

{style_templates[selected_style]}

CRITICAL REQUIREMENTS:
- DO NOT use the phrase "Of course! Could you please share your flight number with me?"
- DO NOT use the phrase "That way, I can help you better"
- DO NOT end with "Thank you!"
- Make your response sound completely different from a standard redirect
- Be creative and natural in your approach
- Acknowledge their specific question about "{last_user_message}"
- Redirect to flight delay compensation in a unique way
- Keep it brief but engaging

Examples of what NOT to say:
- "Of course! Could you please share your flight number with me?"
- "That way, I can help you better with your compensation"
- Generic redirects that sound the same

Be creative and make each response unique!"""),
            ("human", f"Generate a {selected_style} response that's completely different from standard redirects.")
        ])
        
        try:
            # Use a more direct approach with template variation
            redirect_templates = [
                f"I understand you're asking about {last_user_message}, but I'm actually specialized in helping passengers get compensation for flight delays. Let's focus on your flight delay case instead!",
                
                f"That's an interesting question about {last_user_message}! Unfortunately, my expertise is specifically in air passenger rights and flight delay compensation. I'd love to help you with that instead.",
                
                f"I appreciate you asking about {last_user_message}, but I'm here to help with flight delay compensation claims. Let's get back to your delayed flight - that's where I can really help you!",
                
                f"While I can't help with {last_user_message}, I'm actually really good at helping people get compensation for flight delays. Shall we focus on your flight delay case?",
                
                f"I hear you asking about {last_user_message}, but my specialty is flight delay compensation. I'm excited to help you understand your passenger rights and potential compensation!"
            ]
            
            # Select a random template and then use LLM to vary it
            base_template = random.choice(redirect_templates)
            
            variation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are Agent S, a helpful customer service agent for TripFix. 

Take this base response and make it sound more natural and conversational while keeping the same meaning:

Base response: "{base_response}"

Make it sound like a real person talking, not a template. Add personality and make it engaging while keeping the core message the same."""),
                ("human", "Make this response more natural and conversational.")
            ])
            
            # Create a temporary LLM with higher temperature for more varied responses
            from langchain_openai import ChatOpenAI
            varied_llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=self.openai_api_key,
                temperature=0.9  # Even higher temperature for more creativity
            )
            
            chain = variation_prompt | varied_llm
            response = chain.invoke({
                "base_response": base_template
            })
            redirect_message = response.content
        except Exception as e:
            print(f"Error generating redirect message: {e}")
            redirect_message = f"""I understand you're asking about "{last_user_message}", but I'm specifically designed to help with flight delay compensation claims. 

Let's focus on getting your flight delay sorted out first - that's what I do best! 

Could we please continue with the details about your delayed flight? I'm here to help you understand your passenger rights and potential compensation."""
        
        state["messages"].append({
            "role": "assistant",
            "content": redirect_message,
            "timestamp": datetime.now().isoformat(),
            "step": "redirect_on_topic"
        })
        
        return state
    
    def extract_user_info(self, user_message: str) -> Dict[str, Any]:
        """Extract user name and mood from user message using LLM"""
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract user information from the user's message.

Extract:
- user_name: The person's name (first name is fine)
- user_mood: How they're feeling/doing today (e.g., "good", "frustrated", "okay", "tired", "excited")

User message: {user_message}

IMPORTANT: Only extract fields that are clearly mentioned in the user message. Don't make assumptions.
If a field is not mentioned, leave it empty.

Respond with JSON containing only the fields you can extract:
{{
    "user_name": "...",
    "user_mood": "..."
}}"""),
            ("human", "Extract the user information.")
        ])
        
        try:
            chain = extraction_prompt | self.llm
            response = chain.invoke({
                "user_message": user_message
            })
            
            extracted = json.loads(response.content)
            return extracted
            
        except Exception as e:
            print(f"Error extracting user info: {e}")
            return {}
    
    def extract_flight_info(self, user_message: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract flight information from user message using LLM"""
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract flight information from the user's message. 

Current data: {current_data}

Extract and update any of these fields from the user's message:
- flight_number: Flight code (e.g., AC123, WF456)
- flight_date: Date of flight (convert to YYYY-MM-DD format)
- airline: Airline name (if not mentioned, infer from flight code: AC=Air Canada, WF=WestJet, AA=American Airlines, UA=United Airlines, DL=Delta, etc.)
- origin: Departure location (city, airport, or airport code - e.g., "Paris", "CDG", "Paris CDG")
- destination: Arrival location (city, airport, or airport code)
- connecting_airports: Any connecting airports (yes/no and details if yes)
- delay_length: Delay in hours (convert to number)
- delay_reason: Reason given by airline

User message: {user_message}

IMPORTANT: 
- Only extract fields that are clearly mentioned in the user message. Don't make assumptions.
- If a field is already filled in current_data and not mentioned in the user message, keep the existing value.
- For locations, recognize city names, airport codes, and airport names (e.g., "Paris" = origin, "CDG" = origin, "Paris CDG" = origin)
- If the user mentions a location in response to a question about departure, it's likely the origin
- If the user mentions a location in response to a question about arrival, it's likely the destination
- For connecting_airports: Extract "no" or "none" or "direct" as "no connecting flights", "yes" or specific airport names as "yes, [airport details]"
- For airline: If not explicitly mentioned, infer from flight code (AC=Air Canada, WF=WestJet, AA=American Airlines, UA=United Airlines, DL=Delta, BA=British Airways, LH=Lufthansa, AF=Air France, etc.)

Respond with JSON containing only the fields you can extract/update. Leave fields empty if not mentioned.
{{
    "flight_number": "...",
    "flight_date": "...",
    "airline": "...",
    "origin": "...",
    "destination": "...",
    "connecting_airports": "...",
    "delay_length": 0,
    "delay_reason": "..."
}}"""),
            ("human", "Extract the flight information.")
        ])
        
        try:
            chain = extraction_prompt | self.llm
            response = chain.invoke({
                "current_data": json.dumps(current_data, indent=2),
                "user_message": user_message
            })
            
            # Clean and parse JSON response
            response_text = response.content.strip()
            
            # Try to extract JSON from the response if it's embedded in other text
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            
            # Find JSON object boundaries
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]
            
            extracted = json.loads(response_text)
            
            # Merge with existing data, only updating fields that have new values
            updated_data = current_data.copy()
            for key, value in extracted.items():
                if value and str(value).strip() and str(value).strip() != "":
                    # Only update if we have a meaningful value
                    updated_data[key] = value
            
            return updated_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in flight info extraction: {e}")
            print(f"Response content: {response.content if 'response' in locals() else 'No response'}")
            return current_data
        except Exception as e:
            print(f"Error extracting flight info: {e}")
            return current_data
    
    async def process_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Process a user message through the workflow"""
        
        # Get or create session
        session_data = self.database.get_session(session_id)
        if not session_data:
            self.database.create_session(session_id)
            # Initialize new state
            state = IntakeState(
                session_id=session_id,
                messages=[],
                flight_data={},
                current_step="greeting",
                jurisdiction=None,
                jurisdiction_confidence=None,
                jurisdiction_reasoning=None,
                eligibility_result=None,
                eligibility_confidence=None,
                needs_handoff=False,
                handoff_reason=None,
                completed=False,
                next_question=None,
                user_name=None,
                user_mood=None,
                awaiting_feedback=False,
                feedback_iteration=0,
                user_satisfied=None,
                additional_info_provided=False,
                escalation_required=False
            )
        else:
            # Load existing conversation
            conversation = self.database.get_conversation_history(session_id)
            flight_data = json.loads(session_data.get('flight_data') or '{}')
            current_step = session_data.get('status', 'greeting')
            logger.info(f"📊 Loading session {session_id[:8]} with status: {current_step}")
            state = IntakeState(
                session_id=session_id,
                messages=[json.loads(msg['content']) for msg in conversation],
                flight_data=flight_data,
                current_step=current_step,
                jurisdiction=session_data.get('jurisdiction'),
                jurisdiction_confidence=session_data.get('jurisdiction_confidence'),
                jurisdiction_reasoning=None,
                eligibility_result=json.loads(session_data.get('eligibility_result') or 'null'),
                eligibility_confidence=session_data.get('eligibility_confidence'),
                needs_handoff=False,
                handoff_reason=session_data.get('handoff_reason'),
                completed=session_data.get('completed', False),
                next_question=None,
                user_name=flight_data.get('user_name'),
                user_mood=flight_data.get('user_mood'),
                awaiting_feedback=session_data.get('status') == 'awaiting_feedback',
                feedback_iteration=0,
                user_satisfied=None,
                additional_info_provided=False,
                escalation_required=False
            )
        
        # Add user message (only if it's not the "start" message or if it's the first message)
        if user_message != "start" or len(state["messages"]) == 0:
            user_msg = {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            }
            state["messages"].append(user_msg)
            
            # Store user message in database
            self.database.add_message(session_id, "user", json.dumps(user_msg))
        
        # Process the message based on current state
        if state["current_step"] == "greeting" or len(state["messages"]) == 1:
            # First time - show greeting
            result = self.greet_user(state)
        elif state["current_step"] in ["collecting_user_info", "collecting_flight_info", "in_progress"]:
            # Continue with collection
            result = self.collect_flight_info(state)
            # Store user info in flight_data for database persistence
            if result.get("user_name"):
                result["flight_data"]["user_name"] = result["user_name"]
            if result.get("user_mood"):
                result["flight_data"]["user_mood"] = result["user_mood"]
            
            # Continue workflow if all info is collected
            if result["current_step"] == "validated":
                logger.info("📋 All flight info collected, moving to document collection...")
                result = self.collect_supporting_documents(result)
                
                # Continue workflow if documents are collected
                if result["current_step"] == "documents_collected":
                    logger.info("📄 Documents collected, moving to jurisdiction determination...")
                    result = self.determine_jurisdiction(result)
                    
                    # Continue workflow if jurisdiction is determined
                    if result["current_step"] == "jurisdiction_determined":
                        logger.info("🌍 Jurisdiction determined, moving to eligibility assessment...")
                        result = self.assess_eligibility(result)
                        
                        # Complete the workflow
                        if result.get("needs_handoff"):
                            logger.info("🔄 Handoff required, moving to human review...")
                            result = self.handoff_to_human(result)
                        else:
                            logger.info("✅ Auto-processing, completing intake...")
                            result = self.complete_intake(result)
        elif state["current_step"] == "validated":
            # All info collected, move to document collection
            result = self.collect_supporting_documents(state)
            if result["current_step"] == "documents_collected":
                result = self.determine_jurisdiction(result)
                if result["current_step"] == "jurisdiction_determined":
                    result = self.assess_eligibility(result)
                    if result["needs_handoff"]:
                        result = self.handoff_to_human(result)
                    else:
                        result = self.complete_intake(result)
        elif state["current_step"] in ["collecting_documents", "documents_collected"]:
            # Handle document collection
            logger.info(f"📄 Processing document collection step: {state['current_step']}")
            result = self.collect_supporting_documents(state)
            logger.info(f"📄 After document collection, step is: {result['current_step']}")
            if result["current_step"] == "documents_collected":
                logger.info("🌍 Moving to jurisdiction determination...")
                result = self.determine_jurisdiction(result)
                if result["current_step"] == "jurisdiction_determined":
                    logger.info("⚖️ Moving to eligibility assessment...")
                    result = self.assess_eligibility(result)
                    if result["needs_handoff"]:
                        logger.info("🔄 Handoff required, moving to human review...")
                        result = self.handoff_to_human(result)
                    else:
                        logger.info("✅ Auto-processing, completing intake...")
                        result = self.complete_intake(result)
        elif state["current_step"] == "jurisdiction_determined":
            # Jurisdiction determined, move to eligibility assessment
            result = self.assess_eligibility(state)
        elif state["current_step"] == "awaiting_feedback":
            # Handle user feedback after completion
            logger.info(f"🔄 Processing user feedback for session {session_id[:8]}...")
            result = self.handle_user_feedback(state)
            
            # Continue processing based on feedback analysis
            if result["current_step"] == "reopening_analysis":
                # Reopen analysis with additional information
                result = self.reopen_analysis(result)
            elif result["current_step"] == "providing_guidance":
                # Provide guidance to satisfied user
                result = self.provide_guidance(result)
            elif result["current_step"] == "escalating_to_human":
                # Escalate to human
                result = self.handoff_to_human(result)
        elif state["current_step"] in ["guidance_provided", "guidance_complete"]:
            # Handle follow-up questions after guidance has been provided
            logger.info(f"🔄 Processing follow-up question for session {session_id[:8]}...")
            result = self.handle_follow_up_question(state)
        elif state["current_step"] == "completed":
            # Handle follow-up questions even for completed sessions
            logger.info(f"🔄 Processing follow-up question for completed session {session_id[:8]}...")
            result = self.handle_follow_up_question(state)
        else:
            # Already completed or in progress
            return state
        
        # Store assistant messages in database
        for msg in result["messages"]:
            if msg["role"] == "assistant":
                # Check if this message is already in the database
                conversation = self.database.get_conversation_history(session_id)
                message_exists = False
                for existing_msg in conversation:
                    try:
                        existing_content = json.loads(existing_msg['content'])
                        if (existing_content.get('role') == 'assistant' and 
                            existing_content.get('content') == msg.get('content') and
                            existing_content.get('timestamp') == msg.get('timestamp')):
                            message_exists = True
                            break
                    except:
                        continue
                
                if not message_exists:
                    self.database.add_message(session_id, "assistant", json.dumps(msg))
        
        # Determine proper status for database
        db_status = result["current_step"]
        if result.get("needs_handoff", False):
            db_status = "human_review_required"
        elif result.get("completed", False):
            db_status = "completed"
        elif result.get("awaiting_feedback", False):
            db_status = "awaiting_feedback"
        elif result["current_step"] in ["guidance_provided", "guidance_complete"]:
            db_status = "guidance_provided"
        
        # Update session with current flight data and proper status
        completed_value = 1 if result.get("completed", False) else 0
        self.database.update_session(
            session_id,
            flight_data=json.dumps(result["flight_data"]),
            status=db_status,
            completed=completed_value,
            jurisdiction=result.get("jurisdiction"),
            jurisdiction_confidence=result.get("jurisdiction_confidence"),
            eligibility_result=json.dumps(result.get("eligibility_result", {})),
            eligibility_confidence=result.get("eligibility_confidence"),
            handoff_reason=result.get("handoff_reason")
        )
        
        return result
    
    def process_file_upload(self, session_id: str, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process an uploaded supporting file"""
        try:
            # Process the file
            file_info = self.file_processor.process_uploaded_file(file_content, filename, session_id)
            
            if "error" in file_info:
                return {
                    "success": False,
                    "error": file_info["error"],
                    "message": f"Failed to process file: {file_info['error']}"
                }
            
            # Store file in database
            success = self.database.add_supporting_file(
                session_id=session_id,
                filename=file_info["filename"],
                file_type=file_info["file_type"],
                file_size=file_info["file_size"],
                file_path=file_info["file_path"],
                extracted_text=file_info["extracted_text"],
                metadata=file_info["metadata"]
            )
            
            if not success:
                return {
                    "success": False,
                    "error": "Failed to store file in database",
                    "message": "File processed but could not be saved to database"
                }
            
            # Extract flight information from the file
            flight_info = self.file_processor.extract_flight_info(file_info["extracted_text"])
            
            # Update intake progress
            self.database.update_intake_progress(session_id, supporting_files_offered=True)
            
            return {
                "success": True,
                "message": f"File '{filename}' uploaded and processed successfully",
                "file_info": file_info,
                "extracted_flight_info": flight_info,
                "extracted_text_preview": file_info["extracted_text"][:200] + "..." if len(file_info["extracted_text"]) > 200 else file_info["extracted_text"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Error processing file: {str(e)}"
            }
    
    def get_supporting_files(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all supporting files for a session"""
        return self.database.get_supporting_files(session_id)
    
    def get_intake_progress(self, session_id: str) -> Dict[str, Any]:
        """Get intake progress for a session"""
        return self.database.get_intake_progress(session_id) or {}
    
    def is_intake_complete(self, session_id: str) -> bool:
        """Check if intake is complete for a session"""
        return self.database.is_intake_complete(session_id)
    
    def should_await_feedback(self, state: IntakeState) -> str:
        """Determine if we should await user feedback after completion"""
        # Always await feedback after initial completion
        return "await_feedback"
    
    def handle_user_feedback(self, state: IntakeState) -> IntakeState:
        """Handle user feedback after initial analysis"""
        logger.info(f"🔄 Handling user feedback for session {state['session_id'][:8]}...")
        
        # Get the last user message
        last_user_message = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            return state
        
        # Analyze user feedback
        feedback_analysis = self.analyze_user_feedback(last_user_message, state)
        
        # Update feedback iteration
        state["feedback_iteration"] = state.get("feedback_iteration", 0) + 1
        
        # Process feedback based on analysis
        if feedback_analysis["satisfied"]:
            state["user_satisfied"] = True
            # Check if they're asking about next steps
            if feedback_analysis["asking_guidance"]:
                state["current_step"] = "providing_guidance"
            else:
                state["current_step"] = "feedback_complete"
                state["awaiting_feedback"] = False
                state["completed"] = True
        else:
            state["user_satisfied"] = False
            # Check if they're asking for guidance (even if not satisfied)
            if feedback_analysis["asking_guidance"]:
                state["current_step"] = "providing_guidance"
            # Check if they provided additional information
            elif feedback_analysis["additional_info"]:
                state["additional_info_provided"] = True
                state["current_step"] = "reopening_analysis"
            else:
                # Check escalation threshold
                if state["feedback_iteration"] >= 2:
                    state["escalation_required"] = True
                    state["current_step"] = "escalating_to_human"
                else:
                    state["current_step"] = "requesting_clarification"
        

        
        return state
    
    def analyze_user_feedback(self, user_message: str, state: IntakeState) -> Dict[str, Any]:
        """Analyze user feedback to determine satisfaction and intent"""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the user's feedback message to determine their satisfaction and intent.

User message: {user_message}

Context: This is feedback after receiving an eligibility analysis for their flight delay compensation claim.

Analyze and respond with JSON:
{{
    "satisfied": true/false,
    "asking_guidance": true/false,
    "additional_info": true/false,
    "sentiment": "positive/negative/neutral",
    "intent": "satisfied/unsatisfied/asking_questions/providing_info/escalating",
    "key_points": ["list of main points from their message"]
}}

Guidelines:
- "satisfied": true if they seem accepting of the result, false if they disagree or are unhappy
- "asking_guidance": true if they're asking about next steps, process, or how things work
- "additional_info": true if they're providing new information about their case
- Look for phrases like "thank you", "that's helpful", "I understand" for satisfaction
- Look for phrases like "but", "however", "that's not right", "I disagree" for dissatisfaction
- Look for questions about "what happens next", "how does this work", "what should I do" for guidance
- Look for new flight details, additional circumstances, or corrections for additional_info"""),
            ("human", "Analyze this user feedback.")
        ])
        
        try:
            chain = analysis_prompt | self.llm
            response = chain.invoke({"user_message": user_message})
            
            # Parse JSON response
            response_text = response.content.strip()
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]
            
            analysis = json.loads(response_text)
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing user feedback: {e}")
            # Fallback analysis
            message_lower = user_message.lower()
            satisfied = any(phrase in message_lower for phrase in [
                "thank you", "thanks", "that's helpful", "i understand", "okay", "ok", "good", "great"
            ])
            asking_guidance = any(phrase in message_lower for phrase in [
                "what happens next", "what should i do", "how does this work", "next step", "process"
            ])
            additional_info = any(phrase in message_lower for phrase in [
                "but", "however", "actually", "i forgot", "also", "additionally", "one more thing"
            ])
            
            return {
                "satisfied": satisfied,
                "asking_guidance": asking_guidance,
                "additional_info": additional_info,
                "sentiment": "positive" if satisfied else "negative",
                "intent": "satisfied" if satisfied else "unsatisfied",
                "key_points": [user_message[:100]]
            }
    
    def feedback_decision(self, state: IntakeState) -> str:
        """Decide next step based on feedback analysis"""
        if state["escalation_required"]:
            return "escalate"
        elif state["user_satisfied"] and state["current_step"] == "providing_guidance":
            return "guidance"
        elif not state["user_satisfied"] and state["additional_info_provided"]:
            return "reopen"
        elif state["user_satisfied"]:
            return "end"
        else:
            # Request clarification
            return "end"
    
    def reopen_analysis(self, state: IntakeState) -> IntakeState:
        """Reopen analysis with additional information provided by user"""
        logger.info(f"🔄 Reopening analysis for session {state['session_id'][:8]} with additional info...")
        
        # Get the last user message with additional information
        last_user_message = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        # Extract additional flight information
        if last_user_message:
            updated_flight_data = self.extract_flight_info(last_user_message, state["flight_data"])
            state["flight_data"] = updated_flight_data
        
        # Generate reopening message
        reopening_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Agent S, a helpful customer service agent for TripFix.

Generate a message acknowledging the additional information provided and explaining that you're re-analyzing their case.

User Name: {user_name}
Additional Information: {additional_info}

Be:
1. Appreciative of the additional information
2. Reassuring that you're taking it seriously
3. Clear that you're re-analyzing their case
4. Professional and empathetic"""),
            ("human", "Generate a reopening analysis message.")
        ])
        
        try:
            chain = reopening_prompt | self.llm
            response = chain.invoke({
                "user_name": state.get("user_name", "there"),
                "additional_info": last_user_message
            })
            reopening_message = response.content
        except Exception as e:
            logger.error(f"Error generating reopening message: {e}")
            reopening_message = f"Thank you for providing that additional information, {state.get('user_name', 'there')}. I'm going to re-analyze your case with this new information to give you a more accurate assessment."
        
        state["messages"].append({
            "role": "assistant",
            "content": reopening_message,
            "timestamp": datetime.now().isoformat(),
            "step": "reopening_analysis"
        })
        
        # Reset analysis state
        state["current_step"] = "collecting_flight_info"
        state["awaiting_feedback"] = False
        state["jurisdiction"] = None
        state["jurisdiction_confidence"] = None
        state["jurisdiction_reasoning"] = None
        state["eligibility_result"] = None
        state["eligibility_confidence"] = None
        state["needs_handoff"] = False
        state["handoff_reason"] = None
        
        # Update database
        self.database.update_session(
            state["session_id"],
            status="reopening_analysis",
            flight_data=json.dumps(state["flight_data"])
        )
        
        return state
    
    def provide_guidance(self, state: IntakeState) -> IntakeState:
        """Provide guidance to satisfied users about next steps"""
        logger.info(f"📋 Providing guidance for session {state['session_id'][:8]}...")
        
        eligibility = state["eligibility_result"]
        user_name = state.get("user_name", "there")
        
        # Generate comprehensive guidance message
        guidance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Agent S, a helpful customer service agent for TripFix.

Generate a comprehensive guidance message explaining the dispute process and next steps.

User Name: {user_name}
Eligible: {eligible}
Compensation Amount: {compensation_amount}
Jurisdiction: {jurisdiction}

If eligible, explain:
1. How the dispute process works
2. Timeline expectations
3. What happens next
4. How they'll be updated
5. Contact information

If not eligible, explain:
1. Alternative options available
2. How to appeal or request manual review
3. Other potential remedies
4. Contact information for further assistance

Be comprehensive but clear, professional and helpful."""),
            ("human", "Generate guidance message.")
        ])
        
        try:
            chain = guidance_prompt | self.llm
            response = chain.invoke({
                "user_name": user_name,
                "eligible": eligibility["eligible"],
                "compensation_amount": eligibility.get("compensation_amount", 0),
                "jurisdiction": state["jurisdiction"]
            })
            guidance_message = response.content
        except Exception as e:
            logger.error(f"Error generating guidance message: {e}")
            if eligibility["eligible"]:
                guidance_message = f"""📋 **Next Steps for Your Compensation Claim, {user_name}**

**How the Dispute Process Works:**
1. **Documentation Preparation** (1-2 business days)
   - We'll prepare all necessary legal documents
   - Include your flight details and supporting evidence
   
2. **Legal Submission** (3-5 business days)
   - Our legal team submits your claim to the airline
   - Follows {state['jurisdiction']} regulations precisely
   
3. **Airline Response** (15-30 days typically)
   - Airlines have up to 30 days to respond
   - We monitor and follow up if needed
   
4. **Resolution** (30-60 days total)
   - You'll receive compensation directly from the airline
   - We'll keep you updated throughout the process

**What You Can Expect:**
- Email updates at each major milestone
- Direct communication from our legal team if needed
- No upfront costs - we only get paid if you win
- Full transparency on all communications

**Contact Information:**
- Email: legal@tripfix.com
- Phone: 1-800-TRIPFIX
- Case ID: {state['session_id'][:8]}

Thank you for choosing TripFix, {user_name}! We're committed to getting you the ${eligibility.get('compensation_amount', 0):.2f} CAD compensation you deserve. 🛫💰"""
            else:
                guidance_message = f"""📋 **Alternative Options and Next Steps, {user_name}**

**While your case doesn't qualify under current regulations, here are your options:**

**1. Manual Review Request**
- Our senior legal team can review your case manually
- Sometimes there are nuances our automated system misses
- Request at: manual-review@tripfix.com

**2. Alternative Remedies**
- **Travel Insurance Claims**: Check your travel insurance policy
- **Airline Goodwill**: Contact the airline directly for vouchers/credits
- **Credit Card Protection**: Some cards offer travel delay protection

**3. Future Monitoring**
- We can monitor for regulatory changes that might affect your case
- Sign up for updates at: updates@tripfix.com

**4. Other Services**
- We also help with baggage claims, overbooking, and other travel issues
- Visit our website for more information

**Contact Information:**
- General inquiries: help@tripfix.com
- Phone: 1-800-TRIPFIX
- Case ID: {state['session_id'][:8]}

Thank you for reaching out, {user_name}. While we couldn't help with this specific case, we're here if you need assistance with other travel issues! 🛫"""
        
        state["messages"].append({
            "role": "assistant",
            "content": guidance_message,
            "timestamp": datetime.now().isoformat(),
            "step": "guidance_provided"
        })
        
        # Keep session active for follow-up questions
        state["completed"] = False
        state["awaiting_feedback"] = False
        state["current_step"] = "guidance_provided"
        
        # Update database to keep session active for follow-up questions
        self.database.update_session(
            state["session_id"],
            status="guidance_provided",
            completed=False
        )
        
        return state

