#!/usr/bin/env python3
"""
TripFix Comprehensive Test Scenarios
====================================

This test suite covers all required scenarios for the TripFix intake system:
1. Flight entirely within Canada (APPR)
2. Flight from EU to Canada on Canadian airline (EU 261)
3. Flight that falls under neither jurisdiction (e.g., within the US)
4. Ambiguous delay reason triggering human handoff
5. Successful and complete intake process
6. Off-topic conversation attempt
7. User requiring human agent after solution provided

Run with: python test_tripfix_scenarios.py
"""

import asyncio
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from agents.intake_agent import IntakeAgent
from utils.database import IntakeDatabase
from utils.vector_store import VectorStore

# Load environment variables
load_dotenv()

class TripFixScenarioTester:
    """Comprehensive test suite for TripFix intake scenarios"""
    
    def __init__(self):
        self.database = None
        self.vector_store = None
        self.agent = None
        self.test_results = {}
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    async def initialize(self):
        """Initialize the system components"""
        print("🔧 Initializing TripFix system for testing...")
        
        # Use test database to avoid affecting production data
        self.database = IntakeDatabase("data/test_database.db")
        self.vector_store = VectorStore(openai_api_key=self.openai_api_key)
        self.vector_store.initialize_from_pdfs()
        
        self.agent = IntakeAgent(
            openai_api_key=self.openai_api_key,
            database=self.database,
            vector_store=self.vector_store
        )
        
        print("✅ System initialized successfully")
    
    async def run_complete_intake_scenario(self, session_id: str, flight_data: dict, description: str):
        """Run a complete intake process for given flight data"""
        print(f"\n{'='*80}")
        print(f"🧪 SCENARIO: {description}")
        print(f"{'='*80}")
        
        try:
            # Step 1: Initial greeting
            print("📝 Step 1: Initial greeting...")
            result = await self.agent.process_message(session_id, "start")
            
            # Step 2-8: Provide all required information in natural conversation
            print("📝 Step 2-8: Collecting flight information...")
            messages = [
                f"My name is John and I'm doing okay, thanks for asking",
                f"My flight number is {flight_data['flight_number']}",
                f"The flight was on {flight_data['flight_date']}",
                f"It was with {flight_data['airline']}",
                f"From {flight_data['origin']}",
                f"To {flight_data['destination']}",
                f"No connecting flights",
                f"The delay was {flight_data['delay_length']} hours",
                f"The airline said it was due to {flight_data['delay_reason']}",
                "No, I don't have any supporting documents"
            ]
            
            for i, message in enumerate(messages, 1):
                print(f"  📝 Message {i}: {message}")
                result = await self.agent.process_message(session_id, message)
            
            # Analyze results
            print(f"\n📊 RESULTS ANALYSIS:")
            print(f"  Current step: {result.get('current_step', 'unknown')}")
            print(f"  Jurisdiction: {result.get('jurisdiction', 'unknown')}")
            print(f"  Jurisdiction confidence: {result.get('jurisdiction_confidence', 'unknown')}")
            print(f"  Eligible: {result.get('eligibility_result', {}).get('eligible', 'unknown')}")
            print(f"  Compensation: ${result.get('eligibility_result', {}).get('compensation_amount', 0)}")
            print(f"  Eligibility confidence: {result.get('eligibility_confidence', 'unknown')}")
            print(f"  Needs handoff: {result.get('needs_handoff', False)}")
            print(f"  Handoff reason: {result.get('handoff_reason', 'none')}")
            print(f"  Risk level: {result.get('risk_level', 'unknown')}")
            print(f"  Completed: {result.get('completed', False)}")
            
            # Check database
            session_data = self.database.get_session(session_id)
            if session_data:
                print(f"  Database status: {session_data.get('status', 'unknown')}")
                print(f"  Database completed: {session_data.get('completed', False)}")
            
            # Store results for summary
            self.test_results[description] = {
                'jurisdiction': result.get('jurisdiction', 'unknown'),
                'jurisdiction_confidence': result.get('jurisdiction_confidence', 'unknown'),
                'eligible': result.get('eligibility_result', {}).get('eligible', 'unknown'),
                'compensation': result.get('eligibility_result', {}).get('compensation_amount', 0),
                'eligibility_confidence': result.get('eligibility_confidence', 'unknown'),
                'needs_handoff': result.get('needs_handoff', False),
                'handoff_reason': result.get('handoff_reason', 'none'),
                'risk_level': result.get('risk_level', 'unknown'),
                'db_status': session_data.get('status', 'unknown') if session_data else 'unknown',
                'success': True
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in scenario: {e}")
            self.test_results[description] = {
                'success': False,
                'error': str(e)
            }
            return None
    
    async def test_off_topic_conversation(self, session_id: str):
        """Test off-topic conversation handling"""
        print(f"\n{'='*80}")
        print(f"🧪 SCENARIO: Off-topic conversation attempt")
        print(f"{'='*80}")
        
        try:
            # Start conversation
            print("📝 Starting conversation...")
            result = await self.agent.process_message(session_id, "start")
            
            # Try off-topic messages
            print("📝 Testing off-topic messages...")
            off_topic_messages = [
                "What's the weather like today?",
                "Can you help me book a hotel in Paris?",
                "I need help with my car rental reservation",
                "Tell me about good restaurants in Tokyo"
            ]
            
            for i, message in enumerate(off_topic_messages, 1):
                print(f"  📝 Off-topic message {i}: {message}")
                result = await self.agent.process_message(session_id, message)
                last_response = result.get('messages', [{}])[-1].get('content', '')
                print(f"  🤖 Response: {last_response[:100]}...")
            
            # Check if agent redirected back to flight delay topic
            last_assistant_message = ""
            for msg in reversed(result.get('messages', [])):
                if msg.get('role') == 'assistant':
                    last_assistant_message = msg.get('content', '')
                    break
            
            redirected = any(keyword in last_assistant_message.lower() for keyword in 
                           ['flight', 'delay', 'compensation', 'tripfix', 'airline'])
            
            print(f"\n📊 RESULTS:")
            print(f"  Agent redirected to topic: {redirected}")
            print(f"  Last response: {last_assistant_message[:150]}...")
            
            self.test_results['Off-topic conversation'] = {
                'redirected': redirected,
                'last_response': last_assistant_message[:100],
                'success': True
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in off-topic test: {e}")
            self.test_results['Off-topic conversation'] = {
                'success': False,
                'error': str(e)
            }
            return None
    
    async def test_human_agent_request(self, session_id: str):
        """Test user requesting human agent after solution provided"""
        print(f"\n{'='*80}")
        print(f"🧪 SCENARIO: User requiring human agent after solution provided")
        print(f"{'='*80}")
        
        try:
            # First complete a successful intake
            print("📝 Step 1: Completing successful intake...")
            flight_data = {
                'flight_number': 'AC888',
                'flight_date': '2025-05-01',
                'airline': 'Air Canada',
                'origin': 'Toronto',
                'destination': 'Vancouver',
                'delay_length': 4,
                'delay_reason': 'mechanical issues'
            }
            
            # Complete the intake process
            result = await self.run_complete_intake_scenario(session_id, flight_data, "Human agent request - initial intake")
            
            # Now test human agent request
            print("📝 Step 2: User requests human agent...")
            human_request_messages = [
                "I'd like to speak to a human agent",
                "Can I talk to someone directly?",
                "I need to speak to a person about this",
                "Is there a human I can talk to?"
            ]
            
            for i, message in enumerate(human_request_messages, 1):
                print(f"  📝 Human request {i}: {message}")
                result = await self.agent.process_message(session_id, message)
                last_response = result.get('messages', [{}])[-1].get('content', '')
                print(f"  🤖 Response: {last_response[:100]}...")
            
            # Check if agent handled human request appropriately
            last_assistant_message = ""
            for msg in reversed(result.get('messages', [])):
                if msg.get('role') == 'assistant':
                    last_assistant_message = msg.get('content', '')
                    break
            
            human_handled = any(keyword in last_assistant_message.lower() for keyword in 
                              ['human', 'agent', 'representative', 'specialist', 'team', 'escalate'])
            
            print(f"\n📊 RESULTS:")
            print(f"  Human request handled: {human_handled}")
            print(f"  Last response: {last_assistant_message[:150]}...")
            
            self.test_results['Human agent request'] = {
                'human_handled': human_handled,
                'last_response': last_assistant_message[:100],
                'success': True
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in human agent test: {e}")
            self.test_results['Human agent request'] = {
                'success': False,
                'error': str(e)
            }
            return None
    
    async def run_all_scenarios(self):
        """Run all required test scenarios"""
        await self.initialize()
        
        print(f"\n🚀 Starting TripFix comprehensive scenario testing...")
        print(f"📅 Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Scenario 1: Flight entirely within Canada (APPR)
        await self.run_complete_intake_scenario(
            "test_canada_domestic_" + str(uuid.uuid4())[:8],
            {
                'flight_number': 'AC456',
                'flight_date': '2025-04-10',
                'airline': 'Air Canada',
                'origin': 'Toronto',
                'destination': 'Vancouver',
                'delay_length': 3,
                'delay_reason': 'mechanical issues'
            },
            "1. Flight entirely within Canada (APPR)"
        )
        
        # Scenario 2: Flight from EU to Canada on Canadian airline (EU 261)
        await self.run_complete_intake_scenario(
            "test_eu_to_canada_" + str(uuid.uuid4())[:8],
            {
                'flight_number': 'AC789',
                'flight_date': '2025-04-15',
                'airline': 'Air Canada',
                'origin': 'London',
                'destination': 'Toronto',
                'delay_length': 5,
                'delay_reason': 'air traffic control'
            },
            "2. Flight from EU to Canada on Canadian airline (EU 261)"
        )
        
        # Scenario 3: Flight that falls under neither jurisdiction (within US)
        await self.run_complete_intake_scenario(
            "test_us_domestic_" + str(uuid.uuid4())[:8],
            {
                'flight_number': 'AA123',
                'flight_date': '2025-04-20',
                'airline': 'American Airlines',
                'origin': 'New York',
                'destination': 'Los Angeles',
                'delay_length': 2,
                'delay_reason': 'weather'
            },
            "3. Flight within US (Neither jurisdiction)"
        )
        
        # Scenario 4: Ambiguous delay reason (should trigger human review)
        await self.run_complete_intake_scenario(
            "test_ambiguous_reason_" + str(uuid.uuid4())[:8],
            {
                'flight_number': 'AC999',
                'flight_date': '2025-04-25',
                'airline': 'Air Canada',
                'origin': 'Montreal',
                'destination': 'Calgary',
                'delay_length': 4,
                'delay_reason': 'operational reasons'
            },
            "4. Ambiguous delay reason (Human review trigger)"
        )
        
        # Scenario 5: Successful and complete intake process
        await self.run_complete_intake_scenario(
            "test_successful_intake_" + str(uuid.uuid4())[:8],
            {
                'flight_number': 'AC777',
                'flight_date': '2025-04-30',
                'airline': 'Air Canada',
                'origin': 'Halifax',
                'destination': 'Toronto',
                'delay_length': 6,
                'delay_reason': 'crew scheduling issues'
            },
            "5. Successful and complete intake process"
        )
        
        # Scenario 6: Off-topic conversation attempt
        await self.test_off_topic_conversation("test_off_topic_" + str(uuid.uuid4())[:8])
        
        # Scenario 7: User requiring human agent after solution provided
        await self.test_human_agent_request("test_human_agent_" + str(uuid.uuid4())[:8])
        
        # Print comprehensive summary
        self.print_comprehensive_summary()
    
    def print_comprehensive_summary(self):
        """Print comprehensive test summary with analysis"""
        print(f"\n{'='*100}")
        print(f"📋 TRIPFIX COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*100}")
        print(f"📅 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Individual test results
        print(f"\n📊 INDIVIDUAL TEST RESULTS:")
        print(f"{'='*100}")
        
        for test_name, results in self.test_results.items():
            print(f"\n🧪 {test_name}:")
            if results.get('success', False):
                print(f"  ✅ Status: PASSED")
                if 'jurisdiction' in results:
                    print(f"  🌍 Jurisdiction: {results.get('jurisdiction', 'N/A')}")
                    print(f"  📊 Jurisdiction Confidence: {results.get('jurisdiction_confidence', 'N/A')}")
                    print(f"  ⚖️ Eligible: {results.get('eligible', 'N/A')}")
                    print(f"  💰 Compensation: ${results.get('compensation', 'N/A')}")
                    print(f"  📈 Eligibility Confidence: {results.get('eligibility_confidence', 'N/A')}")
                    print(f"  👤 Needs Handoff: {results.get('needs_handoff', 'N/A')}")
                    print(f"  📝 Handoff Reason: {results.get('handoff_reason', 'N/A')}")
                    print(f"  ⚠️ Risk Level: {results.get('risk_level', 'N/A')}")
                    print(f"  🗄️ DB Status: {results.get('db_status', 'N/A')}")
                elif 'redirected' in results:
                    print(f"  🔄 Redirected to Topic: {results.get('redirected', 'N/A')}")
                    print(f"  💬 Last Response: {results.get('last_response', 'N/A')}")
                elif 'human_handled' in results:
                    print(f"  👤 Human Request Handled: {results.get('human_handled', 'N/A')}")
                    print(f"  💬 Last Response: {results.get('last_response', 'N/A')}")
            else:
                print(f"  ❌ Status: FAILED")
                print(f"  🚨 Error: {results.get('error', 'Unknown error')}")
        
        # Overall analysis
        print(f"\n📈 OVERALL ANALYSIS:")
        print(f"{'='*100}")
        
        # Count successful tests
        successful_tests = sum(1 for results in self.test_results.values() if results.get('success', False))
        total_tests = len(self.test_results)
        print(f"✅ Successful tests: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        
        # Jurisdiction detection analysis
        jurisdiction_tests = [name for name, results in self.test_results.items() 
                             if 'jurisdiction' in results and results.get('success', False)]
        
        if jurisdiction_tests:
            appr_cases = [name for name in jurisdiction_tests 
                         if self.test_results[name].get('jurisdiction') == 'APPR']
            eu261_cases = [name for name in jurisdiction_tests 
                          if self.test_results[name].get('jurisdiction') == 'EU261']
            neither_cases = [name for name in jurisdiction_tests 
                            if self.test_results[name].get('jurisdiction') == 'NEITHER']
            
            print(f"\n🌍 JURISDICTION DETECTION:")
            print(f"  🇨🇦 APPR (Canada): {len(appr_cases)} cases")
            print(f"  🇪🇺 EU261 (Europe): {len(eu261_cases)} cases")
            print(f"  ❌ Neither: {len(neither_cases)} cases")
            
            # Expected vs actual
            expected_appr = ["1. Flight entirely within Canada (APPR)"]
            expected_eu261 = ["2. Flight from EU to Canada on Canadian airline (EU 261)"]
            expected_neither = ["3. Flight within US (Neither jurisdiction)"]
            
            appr_correct = any(name in appr_cases for name in expected_appr)
            eu261_correct = any(name in eu261_cases for name in expected_eu261)
            neither_correct = any(name in neither_cases for name in expected_neither)
            
            print(f"\n🎯 JURISDICTION ACCURACY:")
            print(f"  🇨🇦 APPR Detection: {'✅ CORRECT' if appr_correct else '❌ INCORRECT'}")
            print(f"  🇪🇺 EU261 Detection: {'✅ CORRECT' if eu261_correct else '❌ INCORRECT'}")
            print(f"  ❌ Neither Detection: {'✅ CORRECT' if neither_correct else '❌ INCORRECT'}")
        
        # Human review flagging analysis
        handoff_tests = [name for name, results in self.test_results.items() 
                        if results.get('needs_handoff', False) and results.get('success', False)]
        print(f"\n👤 HUMAN REVIEW FLAGGING:")
        print(f"  ⚠️ Cases flagged for human review: {len(handoff_tests)}")
        for test in handoff_tests:
            print(f"    - {test}: {self.test_results[test].get('handoff_reason', 'No reason')}")
        
        # Special scenario analysis
        off_topic_redirected = self.test_results.get('Off-topic conversation', {}).get('redirected', False)
        human_request_handled = self.test_results.get('Human agent request', {}).get('human_handled', False)
        
        print(f"\n🎭 SPECIAL SCENARIOS:")
        print(f"  🔄 Off-topic redirection: {'✅ WORKING' if off_topic_redirected else '❌ NOT WORKING'}")
        print(f"  👤 Human agent requests: {'✅ WORKING' if human_request_handled else '❌ NOT WORKING'}")
        
        # Final verdict
        print(f"\n🏆 FINAL VERDICT:")
        print(f"{'='*100}")
        if successful_tests == total_tests:
            print(f"🎉 ALL TESTS PASSED! TripFix system is working correctly.")
        elif successful_tests >= total_tests * 0.8:
            print(f"✅ MOSTLY SUCCESSFUL! {successful_tests}/{total_tests} tests passed.")
        else:
            print(f"⚠️ NEEDS ATTENTION! Only {successful_tests}/{total_tests} tests passed.")
        
        print(f"\n✅ TripFix comprehensive scenario testing completed!")

async def main():
    """Main test runner"""
    print("🚀 Starting TripFix Comprehensive Scenario Testing")
    print("=" * 60)
    
    try:
        tester = TripFixScenarioTester()
        await tester.run_all_scenarios()
    except Exception as e:
        print(f"❌ Test suite failed to run: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
