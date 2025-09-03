from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from langgraph import Graph, StateGraph, END
from typing import Dict, Any, List, TypedDict, Optional
import json
import uuid
from datetime import datetime

from agents.jurisdiction_agent import JurisdictionAgent
from agents.eligibility_agent import EligibilityAgent
from agents.confidence_scorer import ConfidenceScorer
from utils.database import IntakeDatabase
from utils.vector_store import VectorStore

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
    completed: bool
    next_question: Optional[str]

class IntakeAgent:
    def __init__(self, openai_api_key: str, database: IntakeDatabase, vector_store: VectorStore):
        self.openai_api_key = openai_api_key
        self.database = database
        self.vector_store = vector_store
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            openai_api_key=openai_api_key,
            temperature=0.3
        )
        
        # Initialize specialized agents
        self.jurisdiction_agent = JurisdictionAgent(openai_api_key, vector_store)
        self.eligibility_agent = EligibilityAgent(openai_api_key, vector_store)
        self.confidence_scorer = ConfidenceScorer()
        
        # Required fields for intake
        self.required_fields = [
            'flight_number', 'flight_date', 'airline', 'origin', 
            'destination', 'delay_length', 'delay_reason'
        ]
        
        self.graph = self.create_workflow()
    
    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(IntakeState)
        
        # Add nodes
        workflow.add_node("greet", self.greet_user)
        workflow.add_node("collect_info", self.collect_flight_info)
        workflow.add_node("validate_data", self.validate_flight_data)
        workflow.add_node("determine_jurisdiction", self.determine_jurisdiction)
        workflow.add_node("assess_eligibility", self.assess_eligibility)
        workflow.add_node("score_confidence", self.score_confidence)
        workflow.add_node("handoff_human", self.handoff_to_human)
        workflow.add_node("complete_intake", self.complete_intake)
        workflow.add_node("handle_off_topic", self.handle_off_topic)
        
        # Set entry point
        workflow.set_entry_point("greet")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "collect_info",
            self.should_validate_data,
            {
                "validate": "validate_data",
                "continue_collecting": "collect_info",
                "off_topic": "handle_off_topic"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_data",
            self.data_validation_next,
            {
                "jurisdiction": "determine_jurisdiction",
                "collect_more": "collect_info"
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
        
        # Simple edges
        workflow.add_edge("greet", "collect_info")
        workflow.add_edge("determine_jurisdiction", "assess_eligibility")
        workflow.add_edge("assess_eligibility", "score_confidence")
        workflow.add_edge("handoff_human", END)
        workflow.add_edge("complete_intake", END)
        workflow.add_edge("handle_off_topic", "collect_info")
        
        return workflow.compile()
    
    def greet_user(self, state: IntakeState) -> IntakeState:
        """Initial greeting and setup"""
        greeting = """👋 Welcome to TripFix! I'm here to help you understand your rights and potentially get compensation for your flight delay.

I understand how frustrating flight delays can be, especially when they disrupt important plans. Let's work together to see if you're eligible for compensation under air passenger rights laws.

To get started, I'll need to gather some information about your flight. Can you please tell me your flight number and the date of your delayed flight?"""
        
        state["messages"].append({
            "role": "assistant",
            "content": greeting,
            "timestamp": datetime.now().isoformat()
        })
        
        state["current_step"] = "collecting_flight_info"
        return state
    
    def collect_flight_info(self, state: IntakeState) -> IntakeState:
        """Collect flight information with empathetic, dynamic questions"""
        
        # Determine what information we still need
        missing_fields = []
        for field in self.required_fields:
            if field not in state["flight_data"] or not state["flight_data"][field]:
                missing_fields.append(field)
        
        if not missing_fields:
            return state
        
        # Generate contextual question based on what's missing
        next_field = missing_fields[0]
        previous_data = state["flight_data"]
        
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an empathetic customer service agent for TripFix helping passengers with flight delay compensation.

Generate a natural, conversational question to collect the missing flight information. Be warm, understanding, and professional.

Current conversation context:
- Already collected: {collected_data}
- Next field needed: {next_field}

Field descriptions:
- flight_number: The specific flight number (e.g., AC123, WF456)
- flight_date: Date of the delayed flight
- airline: Name of the airline
- origin: Departure airport/city
- destination: Arrival airport/city  
- delay_length: How many hours the flight was delayed
- delay_reason: What reason the airline gave for the delay

Generate a single, natural question. Be conversational and show empathy for their situation."""),
            ("human", "Generate the next question to ask.")
        ])
        
        try:
            chain = question_prompt | self.llm
            response = chain.invoke({
                "collected_data": json.dumps(previous_data, indent=2),
                "next_field": next_field
            })
            
            question = response.content
        except:
            # Fallback questions
            fallback_questions = {
                'flight_number': "Could you please provide your flight number? It should look something like AC123 or WF456.",
                'flight_date': "What date was your delayed flight? Please provide the original departure date.",
                'airline': "Which airline was operating your flight?",
                'origin': "Which airport or city were you departing from?",
                'destination': "Where were you flying to? Please provide the destination airport or city.",
                'delay_length': "How many hours was your flight delayed? Even an approximate time helps.",
                'delay_reason': "What reason did the airline give you for the delay? Even if it seemed vague, any information helps."
            }
            question = fallback_questions.get(next_field, "Could you provide more details about your flight?")
        
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
        
        if missing_fields:
            return "continue_collecting"
        else:
            return "validate"
    
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
            state["messages"].append({
                "role": "assistant",
                "content": f"I notice some issues with the information provided: {'; '.join(validation_issues)}. Let's clarify these details.",
                "timestamp": datetime.now().isoformat()
            })
        else:
            state["current_step"] = "validated"
        
        return state
    
    def data_validation_next(self, state: IntakeState) -> str:
        """Determine next step after validation"""
        return "jurisdiction" if state["current_step"] == "validated" else "collect_more"
    
    def determine_jurisdiction(self, state: IntakeState) -> IntakeState:
        """Determine applicable jurisdiction"""
        state["messages"].append({
            "role": "assistant",
            "content": "🔍 Let me analyze your flight details to determine which air passenger rights laws apply to your situation...",
            "timestamp": datetime.now().isoformat(),
            "step": "analyzing_jurisdiction"
        })
        
        jurisdiction, reasoning, citations = self.jurisdiction_agent.determine_jurisdiction(
            state["flight_data"]
        )
        
        # Score confidence
        confidence, confidence_explanation = self.confidence_scorer.score_jurisdiction_confidence(
            state["flight_data"], jurisdiction, reasoning
        )
        
        state["jurisdiction"] = jurisdiction
        state["jurisdiction_confidence"] = confidence
        state["jurisdiction_reasoning"] = reasoning
        
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
        state["messages"].append({
            "role": "assistant", 
            "content": f"⚖️ Analyzing your eligibility under {state['jurisdiction']} regulations...",
            "timestamp": datetime.now().isoformat(),
            "step": "analyzing_eligibility"
        })
        
        eligible, compensation, reasoning, citations = self.eligibility_agent.assess_eligibility(
            state["flight_data"], 
            state["jurisdiction"]
        )
        
        state["eligibility_result"] = {
            "eligible": eligible,
            "compensation_amount": compensation,
            "reasoning": reasoning,
            "legal_citations": citations
        }
        
        return state
    
    def score_confidence(self, state: IntakeState) -> IntakeState:
        """Score confidence and determine if handoff needed"""
        
        # Score eligibility confidence
        eligibility_confidence, confidence_explanation = self.confidence_scorer.score_eligibility_confidence(
            state["flight_data"],
            state["eligibility_result"]["legal_citations"]
        )
        
        state["eligibility_confidence"] = eligibility_confidence
        
        # Determine if handoff is needed
        needs_handoff, handoff_reason = self.confidence_scorer.should_handoff_to_human(
            state["jurisdiction_confidence"],
            eligibility_confidence
        )
        
        state["needs_handoff"] = needs_handoff
        state["handoff_reason"] = handoff_reason
        
        # Update database
        self.database.update_session(
            state["session_id"],
            eligibility_result=json.dumps(state["eligibility_result"]),
            eligibility_confidence=eligibility_confidence,
            handoff_reason=handoff_reason if needs_handoff else None
        )
        
        return state
    
    def confidence_decision(self, state: IntakeState) -> str:
        """Decide whether to handoff or complete"""
        return "handoff" if state["needs_handoff"] else "complete"
    
    def handoff_to_human(self, state: IntakeState) -> IntakeState:
        """Hand off to human specialist"""
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
    
    def complete_intake(self, state: IntakeState) -> IntakeState:
        """Complete the automated intake process"""
        eligibility = state["eligibility_result"]
        
        if eligibility["eligible"]:
            completion_message = f"""✅ **Great news! You appear to be eligible for compensation!**

**Compensation Amount:** ${eligibility['compensation_amount']:.2f} CAD

**Legal Basis:** Your case falls under {state['jurisdiction']} regulations. {eligibility['reasoning']}

**Next Steps:**
1. We'll prepare your claim documentation
2. Our legal team will contact the airline on your behalf
3. You'll receive updates as we progress your case

Welcome to TripFix! We're committed to getting you the compensation you deserve. 🛫💰"""
        else:
            completion_message = f"""Unfortunately, based on our analysis, your flight delay doesn't appear to qualify for compensation under applicable {state['jurisdiction']} regulations.

**Reason:** {eligibility['reasoning']}

However, you may still have other options:
- Travel insurance claims
- Airline goodwill gestures
- Vouchers or credits

Our team can still review your case manually if you'd like. Sometimes there are nuances that our automated system might miss."""
        
        state["messages"].append({
            "role": "assistant",
            "content": completion_message,
            "timestamp": datetime.now().isoformat(),
            "step": "intake_complete"
        })
        
        # Update database
        self.database.update_session(
            state["session_id"],
            status="completed",
            completed=True
        )
        
        state["completed"] = True
        return state
    
    def handle_off_topic(self, state: IntakeState) -> IntakeState:
        """Handle off-topic questions"""
        redirect_message = """I understand you might have other questions, but I'm specifically designed to help with flight delay compensation claims. 

Let's focus on getting your flight delay sorted out first - that's what I do best! 

Could we please continue with the details about your delayed flight? I'm here to help you understand your passenger rights and potential compensation."""
        
        state["messages"].append({
            "role": "assistant",
            "content": redirect_message,
            "timestamp": datetime.now().isoformat(),
            "step": "redirect_on_topic"
        })
        
        return state
    
    def extract_flight_info(self, user_message: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract flight information from user message using LLM"""
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract flight information from the user's message. 

Current data: {current_data}

Extract and update any of these fields from the user's message:
- flight_number: Flight code (e.g., AC123, WF456)
- flight_date: Date of flight (convert to YYYY-MM-DD format)
- airline: Airline name
- origin: Departure location
- destination: Arrival location
- delay_length: Delay in hours (convert to number)
- delay_reason: Reason given by airline

User message: {user_message}

Respond with JSON containing only the fields you can extract/update. Leave fields empty if not mentioned.
{{
    "flight_number": "...",
    "flight_date": "...",
    "airline": "...",
    "origin": "...",
    "destination": "...",
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
            
            extracted = json.loads(response.content)
            
            # Merge with existing data
            updated_data = current_data.copy()
            for key, value in extracted.items():
                if value and str(value).strip():
                    updated_data[key] = value
            
            return updated_data
            
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
                next_question=None
            )
        else:
            # Load existing conversation
            conversation = self.database.get_conversation_history(session_id)
            state = IntakeState(
                session_id=session_id,
                messages=[json.loads(msg['content']) for msg in conversation],
                flight_data=json.loads(session_data.get('flight_data', '{}')),
                current_step=session_data.get('status', 'greeting'),
                jurisdiction=session_data.get('jurisdiction'),
                jurisdiction_confidence=session_data.get('jurisdiction_confidence'),
                jurisdiction_reasoning=None,
                eligibility_result=json.loads(session_data.get('eligibility_result', 'null')),
                eligibility_confidence=session_data.get('eligibility_confidence'),
                needs_handoff=False,
                handoff_reason=session_data.get('handoff_reason'),
                completed=session_data.get('completed', False),
                next_question=None
            )
        
        # Add user message
        user_msg = {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        }
        state["messages"].append(user_msg)
        
        # Store user message in database
        self.database.add_message(session_id, "user", json.dumps(user_msg))
        
        # Extract flight info if in collection phase
        if state["current_step"] in ["greeting", "collecting_flight_info"]:
            state["flight_data"] = self.extract_flight_info(user_message, state["flight_data"])
        
        # Run the workflow
        result = self.graph.invoke(state)
        
        # Store assistant messages in database
        for msg in result["messages"]:
            if msg["role"] == "assistant" and msg not in state["messages"]:
                self.database.add_message(session_id, "assistant", json.dumps(msg))
        
        return result


## agents/intake_agent.py (continued)
    def collect_flight_info(self, state: IntakeState) -> IntakeState:
        """Collect flight information with empathetic, dynamic questions"""
        
        # Check if this is initial greeting
        if len(state["messages"]) <= 1:
            return state
        
        # Determine what information we still need
        missing_fields = []
        for field in self.required_fields:
            if field not in state["flight_data"] or not state["flight_data"][field]:
                missing_fields.append(field)
        
        if not missing_fields:
            return state
        
        # Generate contextual question based on what's missing
        next_field = missing_fields[0]
        previous_data = state["flight_data"]
        
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an empathetic customer service agent for TripFix helping passengers with flight delay compensation.

Generate a natural, conversational question to collect the missing flight information. Be warm, understanding, and professional.

Current conversation context:
- Already collected: {collected_data}
- Next field needed: {next_field}

Field descriptions:
- flight_number: The specific flight number (e.g., AC123, WF456)
- flight_date: Date of the delayed flight
- airline: Name of the airline
- origin: Departure airport/city
- destination: Arrival airport/city  
- delay_length: How many hours the flight was delayed
- delay_reason: What reason the airline gave for the delay

Generate a single, natural question. Be conversational and show empathy for their situation."""),
            ("human", "Generate the next question to ask.")
        ])
        
        try:
            chain = question_prompt | self.llm
            response = chain.invoke({
                "collected_data": json.dumps(previous_data, indent=2),
                "next_field": next_field
            })
            
            question = response.content
        except:
            # Fallback questions
            fallback_questions = {
                'flight_number': "Could you please provide your flight number? It should look something like AC123 or WF456.",
                'flight_date': "What date was your delayed flight? Please provide the original departure date.",
                'airline': "Which airline was operating your flight?",
                'origin': "Which airport or city were you departing from?",
                'destination': "Where were you flying to? Please provide the destination airport or city.",
                'delay_length': "How many hours was your flight delayed? Even an approximate time helps.",
                'delay_reason': "What reason did the airline give you for the delay? Even if it seemed vague, any information helps."
            }
            question = fallback_questions.get(next_field, "Could you provide more details about your flight?")
        
        state["messages"].append({
            "role": "assistant", 
            "content": question,
            "timestamp": datetime.now().isoformat(),
            "step": "collecting_info",
            "collecting_field": next_field
        })
        
        return state