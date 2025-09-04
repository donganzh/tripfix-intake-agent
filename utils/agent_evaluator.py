"""
Sophisticated Agent Evaluation Engine for TripFix
Provides comprehensive evaluation metrics for jurisdiction accuracy, confidence calibration,
handoff precision, and end-to-end pipeline testing.
"""

import json
import time
import asyncio
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import tempfile
import shutil
import os
from pathlib import Path

from agents.intake_agent import IntakeAgent
from utils.database import IntakeDatabase
from utils.vector_store import VectorStore


@dataclass
class TestCase:
    """Represents a single test case for evaluation"""
    id: str
    name: str
    description: str
    flight_data: Dict[str, Any]
    expected_jurisdiction: str
    expected_eligible: bool
    expected_compensation: float
    expected_handoff: bool
    difficulty: str  # "easy", "medium", "hard"
    tags: List[str]


@dataclass
class EvaluationResult:
    """Results from evaluating a single test case"""
    test_case_id: str
    actual_jurisdiction: str
    actual_eligible: bool
    actual_compensation: float
    actual_handoff: bool
    jurisdiction_confidence: float
    eligibility_confidence: float
    processing_time: float
    expected_jurisdiction: str = ""
    expected_eligible: bool = False
    expected_handoff: bool = False
    error_message: Optional[str] = None
    
    @property
    def jurisdiction_correct(self) -> bool:
        return self.actual_jurisdiction == self.expected_jurisdiction
    
    @property
    def eligibility_correct(self) -> bool:
        return self.actual_eligible == self.expected_eligible
    
    @property
    def handoff_correct(self) -> bool:
        return self.actual_handoff == self.expected_handoff


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    total_tests: int
    jurisdiction_accuracy: float
    eligibility_accuracy: float
    handoff_precision: float
    handoff_recall: float
    handoff_f1: float
    confidence_calibration_error: float
    average_processing_time: float
    error_rate: float
    
    # Component-specific metrics
    jurisdiction_accuracy_by_type: Dict[str, float]
    confidence_distribution: Dict[str, List[float]]
    performance_by_difficulty: Dict[str, float]


class GoldenTestDataset:
    """Curated test dataset with known correct answers"""
    
    def __init__(self):
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases covering various scenarios"""
        return [
            # Easy Cases - Clear jurisdiction and eligibility
            TestCase(
                id="easy_canadian_domestic",
                name="Canadian Domestic - Clear APPR",
                description="Air Canada domestic flight with clear mechanical delay",
                flight_data={
                    "flight_number": "AC123",
                    "flight_date": "2024-03-15",
                    "airline": "Air Canada",
                    "origin": "Toronto",
                    "destination": "Vancouver",
                    "delay_length": 4.0,
                    "delay_reason": "mechanical issues"
                },
                expected_jurisdiction="APPR",
                expected_eligible=True,
                expected_compensation=1000.0,
                expected_handoff=False,
                difficulty="easy",
                tags=["canadian", "domestic", "mechanical", "clear"]
            ),
            
            TestCase(
                id="easy_eu_departure",
                name="EU Departure - Clear EU261",
                description="Lufthansa flight departing from EU with technical delay",
                flight_data={
                    "flight_number": "LH456",
                    "flight_date": "2024-03-20",
                    "airline": "Lufthansa",
                    "origin": "Frankfurt",
                    "destination": "New York",
                    "delay_length": 5.0,
                    "delay_reason": "technical problems"
                },
                expected_jurisdiction="EU261",
                expected_eligible=True,
                expected_compensation=600.0,
                expected_handoff=False,
                difficulty="easy",
                tags=["eu", "departure", "technical", "clear"]
            ),
            
            TestCase(
                id="easy_us_domestic",
                name="US Domestic - No Jurisdiction",
                description="United domestic flight with no applicable jurisdiction",
                flight_data={
                    "flight_number": "UA789",
                    "flight_date": "2024-03-25",
                    "airline": "United Airlines",
                    "origin": "New York",
                    "destination": "Los Angeles",
                    "delay_length": 3.0,
                    "delay_reason": "weather"
                },
                expected_jurisdiction="NEITHER",
                expected_eligible=False,
                expected_compensation=0.0,
                expected_handoff=False,
                difficulty="easy",
                tags=["us", "domestic", "weather", "no_jurisdiction"]
            ),
            
            # Medium Cases - Some ambiguity
            TestCase(
                id="medium_codeshare",
                name="Code-share Flight - APPR",
                description="WestJet marketed, Air Canada operated flight",
                flight_data={
                    "flight_number": "WS123",
                    "flight_date": "2024-04-01",
                    "airline": "WestJet operated by Air Canada",
                    "origin": "Calgary",
                    "destination": "Toronto",
                    "delay_length": 3.5,
                    "delay_reason": "crew scheduling"
                },
                expected_jurisdiction="APPR",
                expected_eligible=True,
                expected_compensation=1000.0,
                expected_handoff=True,  # Code-share complexity
                difficulty="medium",
                tags=["canadian", "codeshare", "crew", "ambiguous"]
            ),
            
            TestCase(
                id="medium_borderline_delay",
                name="Borderline Delay Duration",
                description="Delay just over 3-hour threshold",
                flight_data={
                    "flight_number": "AC456",
                    "flight_date": "2024-04-05",
                    "airline": "Air Canada",
                    "origin": "Montreal",
                    "destination": "Toronto",
                    "delay_length": 3.1,
                    "delay_reason": "operational requirements"
                },
                expected_jurisdiction="APPR",
                expected_eligible=True,
                expected_compensation=1000.0,
                expected_handoff=True,  # Borderline case
                difficulty="medium",
                tags=["canadian", "borderline", "operational", "threshold"]
            ),
            
            TestCase(
                id="medium_weather_ambiguity",
                name="Weather Delay - EU261",
                description="Weather delay that may or may not be extraordinary",
                flight_data={
                    "flight_number": "AF789",
                    "flight_date": "2024-04-10",
                    "airline": "Air France",
                    "origin": "Paris",
                    "destination": "London",
                    "delay_length": 4.0,
                    "delay_reason": "weather conditions"
                },
                expected_jurisdiction="EU261",
                expected_eligible=True,
                expected_compensation=250.0,
                expected_handoff=True,  # Weather ambiguity
                difficulty="medium",
                tags=["eu", "weather", "ambiguous", "extraordinary"]
            ),
            
            # Hard Cases - Complex scenarios
            TestCase(
                id="hard_multi_jurisdiction",
                name="Multi-Jurisdiction Route",
                description="Complex route with potential multiple jurisdictions",
                flight_data={
                    "flight_number": "AC999",
                    "flight_date": "2024-04-15",
                    "airline": "Air Canada",
                    "origin": "Toronto",
                    "destination": "Paris",
                    "delay_length": 6.0,
                    "delay_reason": "operational reasons"
                },
                expected_jurisdiction="APPR",  # Canadian airline
                expected_eligible=True,
                expected_compensation=1000.0,
                expected_handoff=True,  # Multi-jurisdiction complexity
                difficulty="hard",
                tags=["multi_jurisdiction", "international", "operational", "complex"]
            ),
            
            TestCase(
                id="hard_extraordinary_circumstances",
                name="Extraordinary Circumstances",
                description="Severe weather that qualifies as extraordinary",
                flight_data={
                    "flight_number": "LH888",
                    "flight_date": "2024-04-20",
                    "airline": "Lufthansa",
                    "origin": "Munich",
                    "destination": "Berlin",
                    "delay_length": 8.0,
                    "delay_reason": "severe weather conditions"
                },
                expected_jurisdiction="EU261",
                expected_eligible=False,  # Extraordinary circumstances
                expected_compensation=0.0,
                expected_handoff=True,  # Complex legal determination
                difficulty="hard",
                tags=["eu", "extraordinary", "severe_weather", "complex"]
            ),
            
            TestCase(
                id="hard_missing_data",
                name="Incomplete Information",
                description="Case with missing critical information",
                flight_data={
                    "flight_number": "AC777",
                    "flight_date": "2024-04-25",
                    "airline": "Air Canada",
                    "origin": "Vancouver",
                    "destination": "Toronto",
                    "delay_length": 4.0,
                    "delay_reason": ""  # Missing delay reason
                },
                expected_jurisdiction="APPR",
                expected_eligible=True,
                expected_compensation=1000.0,
                expected_handoff=True,  # Missing data requires review
                difficulty="hard",
                tags=["canadian", "missing_data", "incomplete", "complex"]
            ),
            
            # Edge Cases
            TestCase(
                id="edge_very_short_delay",
                name="Very Short Delay",
                description="Delay under compensation threshold",
                flight_data={
                    "flight_number": "AC555",
                    "flight_date": "2024-05-01",
                    "airline": "Air Canada",
                    "origin": "Ottawa",
                    "destination": "Toronto",
                    "delay_length": 2.5,
                    "delay_reason": "air traffic control"
                },
                expected_jurisdiction="APPR",
                expected_eligible=False,  # Under threshold
                expected_compensation=0.0,
                expected_handoff=False,
                difficulty="easy",
                tags=["canadian", "short_delay", "under_threshold", "clear"]
            ),
            
            TestCase(
                id="edge_very_long_delay",
                name="Very Long Delay",
                description="Extremely long delay with high compensation",
                flight_data={
                    "flight_number": "AF444",
                    "flight_date": "2024-05-05",
                    "airline": "Air France",
                    "origin": "Lyon",
                    "destination": "Paris",
                    "delay_length": 12.0,
                    "delay_reason": "mechanical failure"
                },
                expected_jurisdiction="EU261",
                expected_eligible=True,
                expected_compensation=600.0,
                expected_handoff=True,  # High-value claim
                difficulty="medium",
                tags=["eu", "long_delay", "high_value", "mechanical"]
            ),
            
            # Jurisdiction Edge Cases
            TestCase(
                id="edge_eu_arrival",
                name="EU Arrival - EU261",
                description="Non-EU airline arriving in EU",
                flight_data={
                    "flight_number": "AC333",
                    "flight_date": "2024-05-10",
                    "airline": "Air Canada",
                    "origin": "Toronto",
                    "destination": "Amsterdam",
                    "delay_length": 4.0,
                    "delay_reason": "technical issues"
                },
                expected_jurisdiction="EU261",  # EU arrival
                expected_eligible=True,
                expected_compensation=600.0,
                expected_handoff=False,
                difficulty="medium",
                tags=["eu_arrival", "canadian_airline", "technical", "clear"]
            ),
            
            TestCase(
                id="edge_third_country",
                name="Third Country Route",
                description="Route between two non-EU/Canada countries",
                flight_data={
                    "flight_number": "UA222",
                    "flight_date": "2024-05-15",
                    "airline": "United Airlines",
                    "origin": "Tokyo",
                    "destination": "Sydney",
                    "delay_length": 5.0,
                    "delay_reason": "operational reasons"
                },
                expected_jurisdiction="NEITHER",
                expected_eligible=False,
                expected_compensation=0.0,
                expected_handoff=False,
                difficulty="easy",
                tags=["third_country", "no_jurisdiction", "operational", "clear"]
            ),
            
            # Confidence Testing Cases
            TestCase(
                id="confidence_high_certainty",
                name="High Certainty Case",
                description="Case with very clear jurisdiction and eligibility",
                flight_data={
                    "flight_number": "AC111",
                    "flight_date": "2024-05-20",
                    "airline": "Air Canada",
                    "origin": "Halifax",
                    "destination": "Toronto",
                    "delay_length": 4.0,
                    "delay_reason": "mechanical problems"
                },
                expected_jurisdiction="APPR",
                expected_eligible=True,
                expected_compensation=1000.0,
                expected_handoff=False,
                difficulty="easy",
                tags=["high_confidence", "clear", "mechanical", "canadian"]
            ),
            
            TestCase(
                id="confidence_low_certainty",
                name="Low Certainty Case",
                description="Case with ambiguous information requiring human review",
                flight_data={
                    "flight_number": "WS666",
                    "flight_date": "2024-05-25",
                    "airline": "WestJet",
                    "origin": "Edmonton",
                    "destination": "Toronto",
                    "delay_length": 3.0,
                    "delay_reason": "operational reasons"  # Vague reason
                },
                expected_jurisdiction="APPR",
                expected_eligible=True,
                expected_compensation=1000.0,
                expected_handoff=True,  # Ambiguous reason
                difficulty="hard",
                tags=["low_confidence", "ambiguous", "operational", "complex"]
            )
        ]
    
    def get_test_cases_by_difficulty(self, difficulty: str) -> List[TestCase]:
        """Get test cases filtered by difficulty level"""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]
    
    def get_test_cases_by_tag(self, tag: str) -> List[TestCase]:
        """Get test cases filtered by tag"""
        return [tc for tc in self.test_cases if tag in tc.tags]
    
    def get_all_test_cases(self) -> List[TestCase]:
        """Get all test cases"""
        return self.test_cases


class AgentEvaluator:
    """Sophisticated evaluation engine for TripFix agents"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.test_dataset = GoldenTestDataset()
        self.results: List[EvaluationResult] = []
        
    async def evaluate_single_case(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case"""
        start_time = time.time()
        
        # Create temporary environment for testing
        temp_db = tempfile.mkdtemp()
        temp_vector = tempfile.mkdtemp()
        
        try:
            # Initialize components
            database = IntakeDatabase(os.path.join(temp_db, "test.db"))
            vector_store = VectorStore(temp_vector, self.openai_api_key)
            agent = IntakeAgent(
                openai_api_key=self.openai_api_key,
                database=database,
                vector_store=vector_store
            )
            
            # Process the test case
            session_id = f"eval_{test_case.id}"
            
            # Simulate conversation to collect all flight data
            messages = [
                "Hello, I had a delayed flight",
                f"{test_case.flight_data['airline']} {test_case.flight_data['flight_number']} from {test_case.flight_data['origin']} to {test_case.flight_data['destination']} on {test_case.flight_data['flight_date']}",
                f"The flight was delayed {test_case.flight_data['delay_length']} hours due to {test_case.flight_data['delay_reason']}"
            ]
            
            result = None
            for message in messages:
                result = await agent.process_message(session_id, message)
            
            processing_time = time.time() - start_time
            
            # Extract results
            return EvaluationResult(
                test_case_id=test_case.id,
                actual_jurisdiction=result.get("jurisdiction", "UNKNOWN"),
                actual_eligible=result.get("eligibility_result", {}).get("eligible", False),
                actual_compensation=result.get("eligibility_result", {}).get("compensation_amount", 0.0),
                actual_handoff=result.get("needs_handoff", False),
                jurisdiction_confidence=result.get("jurisdiction_confidence", 0.0),
                eligibility_confidence=result.get("eligibility_confidence", 0.0),
                processing_time=processing_time,
                expected_jurisdiction=test_case.expected_jurisdiction,
                expected_eligible=test_case.expected_eligible,
                expected_handoff=test_case.expected_handoff
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return EvaluationResult(
                test_case_id=test_case.id,
                actual_jurisdiction="ERROR",
                actual_eligible=False,
                actual_compensation=0.0,
                actual_handoff=False,
                jurisdiction_confidence=0.0,
                eligibility_confidence=0.0,
                processing_time=processing_time,
                expected_jurisdiction=test_case.expected_jurisdiction,
                expected_eligible=test_case.expected_eligible,
                expected_handoff=test_case.expected_handoff,
                error_message=str(e)
            )
        finally:
            # Cleanup
            shutil.rmtree(temp_db, ignore_errors=True)
            shutil.rmtree(temp_vector, ignore_errors=True)
    
    async def evaluate_all_cases(self, test_cases: Optional[List[TestCase]] = None) -> List[EvaluationResult]:
        """Evaluate all test cases or a subset"""
        if test_cases is None:
            test_cases = self.test_dataset.get_all_test_cases()
        
        results = []
        for test_case in test_cases:
            print(f"Evaluating: {test_case.name}")
            result = await self.evaluate_single_case(test_case)
            results.append(result)
        
        self.results = results
        return results
    
    def calculate_metrics(self) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics"""
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_all_cases() first.")
        
        # Basic accuracy metrics
        total_tests = len(self.results)
        jurisdiction_correct = sum(1 for r in self.results if r.jurisdiction_correct)
        eligibility_correct = sum(1 for r in self.results if r.eligibility_correct)
        handoff_correct = sum(1 for r in self.results if r.handoff_correct)
        
        jurisdiction_accuracy = jurisdiction_correct / total_tests
        eligibility_accuracy = eligibility_correct / total_tests
        
        # Handoff precision/recall/F1
        true_positives = sum(1 for r in self.results if r.actual_handoff and r.expected_handoff)
        false_positives = sum(1 for r in self.results if r.actual_handoff and not r.expected_handoff)
        false_negatives = sum(1 for r in self.results if not r.actual_handoff and r.expected_handoff)
        
        handoff_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        handoff_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        handoff_f1 = 2 * (handoff_precision * handoff_recall) / (handoff_precision + handoff_recall) if (handoff_precision + handoff_recall) > 0 else 0
        
        # Confidence calibration error
        confidence_calibration_error = self._calculate_calibration_error()
        
        # Performance metrics
        processing_times = [r.processing_time for r in self.results]
        average_processing_time = statistics.mean(processing_times)
        
        error_rate = sum(1 for r in self.results if r.error_message is not None) / total_tests
        
        # Component-specific metrics
        jurisdiction_accuracy_by_type = self._calculate_jurisdiction_accuracy_by_type()
        confidence_distribution = self._calculate_confidence_distribution()
        performance_by_difficulty = self._calculate_performance_by_difficulty()
        
        return EvaluationMetrics(
            total_tests=total_tests,
            jurisdiction_accuracy=jurisdiction_accuracy,
            eligibility_accuracy=eligibility_accuracy,
            handoff_precision=handoff_precision,
            handoff_recall=handoff_recall,
            handoff_f1=handoff_f1,
            confidence_calibration_error=confidence_calibration_error,
            average_processing_time=average_processing_time,
            error_rate=error_rate,
            jurisdiction_accuracy_by_type=jurisdiction_accuracy_by_type,
            confidence_distribution=confidence_distribution,
            performance_by_difficulty=performance_by_difficulty
        )
    
    def _calculate_calibration_error(self) -> float:
        """Calculate confidence calibration error (ECE - Expected Calibration Error)"""
        # Group results by confidence bins
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        total_error = 0.0
        
        for bin_min, bin_max in bins:
            bin_results = [r for r in self.results if bin_min <= r.jurisdiction_confidence < bin_max]
            if not bin_results:
                continue
            
            bin_accuracy = sum(1 for r in bin_results if r.jurisdiction_correct) / len(bin_results)
            bin_confidence = statistics.mean([r.jurisdiction_confidence for r in bin_results])
            
            total_error += abs(bin_accuracy - bin_confidence) * len(bin_results)
        
        return total_error / len(self.results) if self.results else 0.0
    
    def _calculate_jurisdiction_accuracy_by_type(self) -> Dict[str, float]:
        """Calculate accuracy by jurisdiction type"""
        accuracy_by_type = {}
        
        # Get expected jurisdictions from test cases
        test_case_map = {tc.id: tc for tc in self.test_dataset.get_all_test_cases()}
        
        for jurisdiction in ["APPR", "EU261", "NEITHER"]:
            jurisdiction_results = [
                r for r in self.results 
                if test_case_map.get(r.test_case_id, {}).expected_jurisdiction == jurisdiction
            ]
            
            if jurisdiction_results:
                correct = sum(1 for r in jurisdiction_results if r.jurisdiction_correct)
                accuracy_by_type[jurisdiction] = correct / len(jurisdiction_results)
            else:
                accuracy_by_type[jurisdiction] = 0.0
        
        return accuracy_by_type
    
    def _calculate_confidence_distribution(self) -> Dict[str, List[float]]:
        """Calculate confidence score distributions"""
        return {
            "jurisdiction_confidence": [r.jurisdiction_confidence for r in self.results],
            "eligibility_confidence": [r.eligibility_confidence for r in self.results]
        }
    
    def _calculate_performance_by_difficulty(self) -> Dict[str, float]:
        """Calculate performance metrics by difficulty level"""
        performance_by_difficulty = {}
        
        # Get test case difficulties
        test_case_map = {tc.id: tc for tc in self.test_dataset.get_all_test_cases()}
        
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_results = [
                r for r in self.results 
                if test_case_map.get(r.test_case_id, {}).difficulty == difficulty
            ]
            
            if difficulty_results:
                accuracy = sum(1 for r in difficulty_results if r.jurisdiction_correct) / len(difficulty_results)
                performance_by_difficulty[difficulty] = accuracy
            else:
                performance_by_difficulty[difficulty] = 0.0
        
        return performance_by_difficulty
    
    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        if not self.results:
            return "No evaluation results available."
        
        metrics = self.calculate_metrics()
        
        report = f"""
# TripFix Agent Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Tests**: {metrics.total_tests}
- **Jurisdiction Accuracy**: {metrics.jurisdiction_accuracy:.2%}
- **Eligibility Accuracy**: {metrics.eligibility_accuracy:.2%}
- **Handoff F1 Score**: {metrics.handoff_f1:.3f}
- **Confidence Calibration Error**: {metrics.confidence_calibration_error:.3f}
- **Average Processing Time**: {metrics.average_processing_time:.2f}s
- **Error Rate**: {metrics.error_rate:.2%}

## Component Performance

### Jurisdiction Accuracy by Type
"""
        
        for jurisdiction, accuracy in metrics.jurisdiction_accuracy_by_type.items():
            report += f"- **{jurisdiction}**: {accuracy:.2%}\n"
        
        report += f"""
### Performance by Difficulty
"""
        
        for difficulty, accuracy in metrics.performance_by_difficulty.items():
            report += f"- **{difficulty.title()}**: {accuracy:.2%}\n"
        
        report += f"""
## Detailed Results

### Failed Cases
"""
        
        failed_cases = [r for r in self.results if not r.jurisdiction_correct or r.error_message]
        for result in failed_cases:
            test_case = next(tc for tc in self.test_dataset.get_all_test_cases() if tc.id == result.test_case_id)
            report += f"- **{test_case.name}**: Expected {test_case.expected_jurisdiction}, got {result.actual_jurisdiction}"
            if result.error_message:
                report += f" (Error: {result.error_message})"
            report += "\n"
        
        return report
    
    def save_results(self, filepath: str):
        """Save evaluation results to JSON file"""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "test_cases": [asdict(tc) for tc in self.test_dataset.get_all_test_cases()],
            "results": [asdict(r) for r in self.results],
            "metrics": asdict(self.calculate_metrics())
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def load_results(self, filepath: str):
        """Load evaluation results from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct results
        self.results = [EvaluationResult(**result_data) for result_data in data["results"]]


# Convenience functions for easy evaluation
async def run_full_evaluation(openai_api_key: str) -> EvaluationMetrics:
    """Run full evaluation suite and return metrics"""
    evaluator = AgentEvaluator(openai_api_key)
    await evaluator.evaluate_all_cases()
    return evaluator.calculate_metrics()


async def run_quick_evaluation(openai_api_key: str, difficulty: str = "easy") -> EvaluationMetrics:
    """Run quick evaluation on subset of test cases"""
    evaluator = AgentEvaluator(openai_api_key)
    test_cases = evaluator.test_dataset.get_test_cases_by_difficulty(difficulty)
    await evaluator.evaluate_all_cases(test_cases)
    return evaluator.calculate_metrics()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        evaluator = AgentEvaluator(os.getenv("OPENAI_API_KEY"))
        print("Running full evaluation...")
        await evaluator.evaluate_all_cases()
        
        metrics = evaluator.calculate_metrics()
        print(f"Jurisdiction Accuracy: {metrics.jurisdiction_accuracy:.2%}")
        print(f"Eligibility Accuracy: {metrics.eligibility_accuracy:.2%}")
        print(f"Handoff F1: {metrics.handoff_f1:.3f}")
        
        # Save results
        evaluator.save_results("evaluation_results.json")
        print("Results saved to evaluation_results.json")
    
    asyncio.run(main())
