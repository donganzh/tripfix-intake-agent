import pytest
import asyncio
from agents.intake_agent import IntakeAgent
from utils.database import IntakeDatabase
from utils.vector_store import VectorStore
import os

class TestTripFixIntake:
    
    @pytest.fixture
    def setup_system(self):
        """Setup test environment"""
        database = IntakeDatabase("test_database.db")
        vector_store = VectorStore("test_vectorstore", os.getenv("OPENAI_API_KEY"))
        agent = IntakeAgent(os.getenv("OPENAI_API_KEY"), database, vector_store)
        return agent, database
    
    def test_canadian_domestic_flight(self, setup_system):
        """Test Case 1: Flight entirely within Canada (APPR)"""
        agent, database = setup_system
        session_id = "test_canadian_flight"
        
        # Simulate conversation
        messages = [
            "My flight AC123 from Toronto to Vancouver on March 15th was delayed 4 hours due to crew scheduling issues",
        ]
        
        for message in messages:
            result = asyncio.run(agent.process_message(session_id, message))
        
        session_data = database.get_session(session_id)
        assert session_data["jurisdiction"] == "EU261"
    
    def test_no_jurisdiction_flight(self, setup_system):
        """Test Case 3: Flight with no applicable jurisdiction (US domestic)"""
        agent, database = setup_system
        session_id = "test_us_domestic"
        
        messages = [
            "My United flight UA456 from New York JFK to Los Angeles LAX on March 10th was delayed 5 hours due to air traffic control"
        ]
        
        for message in messages:
            result = asyncio.run(agent.process_message(session_id, message))
        
        session_data = database.get_session(session_id)
        assert session_data["jurisdiction"] == "NEITHER"
    
    def test_ambiguous_delay_reason(self, setup_system):
        """Test Case 4: Ambiguous delay reason triggering human handoff"""
        agent, database = setup_system
        session_id = "test_ambiguous_delay"
        
        messages = [
            "My WestJet flight WS789 from Calgary to Toronto on March 25th was delayed 6 hours",
            "The airline just said 'operational reasons' - nothing more specific"
        ]
        
        for message in messages:
            result = asyncio.run(agent.process_message(session_id, message))
        
        session_data = database.get_session(session_id)
        assert session_data["status"] == "human_review_required"
        assert "ambiguous" in session_data["handoff_reason"].lower()
    
    def test_complete_intake_process(self, setup_system):
        """Test Case 5: Complete successful intake"""
        agent, database = setup_system
        session_id = "test_complete_intake"
        
        messages = [
            "Hi, I had a delayed flight",
            "Flight AC100 from Toronto to Montreal on March 30th",
            "Air Canada",
            "The flight was delayed 3 hours",
            "They said it was due to crew scheduling problems"
        ]
        
        for message in messages:
            result = asyncio.run(agent.process_message(session_id, message))
        
        session_data = database.get_session(session_id)
        assert session_data["completed"] == True
        assert session_data["jurisdiction"] in ["APPR", "EU261", "NEITHER"]
    
    def test_off_topic_conversation(self, setup_system):
        """Test Case 6: Off-topic conversation handling"""
        agent, database = setup_system
        session_id = "test_off_topic"
        
        messages = [
            "What's the weather like in Hawaii?",
            "Can you recommend good restaurants?"
        ]
        
        for message in messages:
            result = asyncio.run(agent.process_message(session_id, message))
        
        # Check that agent redirected back to flight topics
        conversation = database.get_conversation_history(session_id)
        assistant_messages = [msg for msg in conversation if json.loads(msg['content'])['role'] == 'assistant']
        
        # Should contain redirection messages
        assert any('flight' in json.loads(msg['content'])['content'].lower() for msg in assistant_messages)

if __name__ == "__main__":
    pytest.main([__file__])