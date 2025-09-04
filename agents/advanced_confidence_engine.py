from typing import Dict, Any, Tuple, List, Optional
import re
import json
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging for agents
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"          # 85%+ confidence - Auto-process
    MEDIUM = "medium"    # 70-85% confidence - Review within 24 hours
    HIGH = "high"        # 50-70% confidence - Priority review within 1 hour
    CRITICAL = "critical" # <50% confidence - Immediate human intervention

@dataclass
class RiskFactor:
    name: str
    weight: float
    score: float
    reasoning: str
    multiplier: float = 1.0

@dataclass
class RiskAssessment:
    overall_confidence: float
    risk_level: RiskLevel
    risk_factors: List[RiskFactor]
    handoff_required: bool
    handoff_priority: str
    reasoning: str
    patterns_detected: List[str]
    contextual_factors: List[str]

class AdvancedConfidenceEngine:
    def __init__(self):
        # Risk factor weights (must sum to 1.0)
        self.risk_weights = {
            'jurisdiction_clarity': 0.25,
            'legal_complexity': 0.20,
            'delay_reason_ambiguity': 0.20,
            'data_completeness': 0.15,
            'regulatory_edge_cases': 0.10,
            'financial_impact': 0.05,
            'precedent_similarity': 0.05
        }
        
        # Risk level thresholds - adjusted for more reasonable auto-processing
        self.risk_thresholds = {
            RiskLevel.LOW: 0.75,      # Lowered from 0.85 - more cases auto-process
            RiskLevel.MEDIUM: 0.60,   # Lowered from 0.70
            RiskLevel.HIGH: 0.40,     # Lowered from 0.50
            RiskLevel.CRITICAL: 0.0
        }
        
        # Pattern detection rules
        self.multi_jurisdiction_indicators = [
            'paris', 'london', 'new york', 'frankfurt', 'amsterdam', 'brussels', 'dublin', 'zurich'
        ]
        
        self.code_share_indicators = [
            'operated by', 'marketed by', 'code share', 'codeshare'
        ]
        
        self.ambiguous_delay_reasons = [
            'operational reasons', 'technical issues', 'crew scheduling',
            'operational requirements', 'network optimization', 'unforeseen circumstances',
            'air traffic control', 'airport congestion', 'ground handling'
        ]
        
        self.extraordinary_circumstances = [
            'weather', 'strike', 'security', 'terrorism', 'political unrest',
            'natural disaster', 'medical emergency', 'bird strike'
        ]
        
        self.uncertainty_indicators = [
            'i think', 'around', 'approximately', 'maybe', 'possibly',
            'not sure', 'unclear', 'don\'t remember', 'might have been'
        ]

    def assess_risk(self, 
                   flight_data: Dict[str, Any],
                   jurisdiction_result: str,
                   jurisdiction_reasoning: str,
                   eligibility_result: Dict[str, Any],
                   conversation_history: List[Dict[str, Any]] = None) -> RiskAssessment:
        """Perform comprehensive multi-factor risk assessment"""
        
        risk_factors = []
        patterns_detected = []
        contextual_factors = []
        
        # 1. Jurisdiction Clarity (25%)
        jurisdiction_factor = self._assess_jurisdiction_clarity(
            flight_data, jurisdiction_result, jurisdiction_reasoning
        )
        risk_factors.append(jurisdiction_factor)
        
        # 2. Legal Complexity (20%)
        legal_complexity_factor = self._assess_legal_complexity(
            flight_data, eligibility_result, jurisdiction_result
        )
        risk_factors.append(legal_complexity_factor)
        
        # 3. Delay Reason Ambiguity (20%)
        delay_ambiguity_factor = self._assess_delay_reason_ambiguity(
            flight_data, eligibility_result
        )
        risk_factors.append(delay_ambiguity_factor)
        
        # 4. Data Completeness (15%)
        data_completeness_factor = self._assess_data_completeness(
            flight_data, eligibility_result
        )
        risk_factors.append(data_completeness_factor)
        
        # 5. Regulatory Edge Cases (10%)
        regulatory_edge_factor = self._assess_regulatory_edge_cases(
            flight_data, jurisdiction_result, eligibility_result
        )
        risk_factors.append(regulatory_edge_factor)
        
        # 6. Financial Impact (5%)
        financial_impact_factor = self._assess_financial_impact(
            flight_data, eligibility_result
        )
        risk_factors.append(financial_impact_factor)
        
        # 7. Precedent Similarity (5%)
        precedent_factor = self._assess_precedent_similarity(
            flight_data, eligibility_result
        )
        risk_factors.append(precedent_factor)
        
        # Pattern detection
        patterns_detected = self._detect_patterns(flight_data, eligibility_result)
        
        # Contextual factors from conversation
        if conversation_history:
            contextual_factors = self._analyze_conversation_context(conversation_history)
        
        # Calculate weighted overall confidence
        overall_confidence = self._calculate_weighted_confidence(risk_factors)
        
        # Determine risk level and handoff requirements
        risk_level = self._determine_risk_level(overall_confidence)
        handoff_required, handoff_priority = self._determine_handoff_requirements(
            risk_level, risk_factors, patterns_detected
        )
        
        # Generate comprehensive reasoning
        reasoning = self._generate_reasoning(risk_factors, patterns_detected, contextual_factors)
        
        return RiskAssessment(
            overall_confidence=overall_confidence,
            risk_level=risk_level,
            risk_factors=risk_factors,
            handoff_required=handoff_required,
            handoff_priority=handoff_priority,
            reasoning=reasoning,
            patterns_detected=patterns_detected,
            contextual_factors=contextual_factors
        )

    def _assess_jurisdiction_clarity(self, flight_data: Dict[str, Any], 
                                   jurisdiction_result: str, reasoning: str) -> RiskFactor:
        """Assess jurisdiction clarity with multi-jurisdiction detection"""
        score = 1.0
        reasoning_parts = []
        multiplier = 1.0
        
        origin = flight_data.get('origin', '').lower()
        destination = flight_data.get('destination', '').lower()
        airline = flight_data.get('airline', '').lower()
        
        # Multi-jurisdiction route detection
        if any(indicator in origin or indicator in destination 
               for indicator in self.multi_jurisdiction_indicators):
            score *= 0.85
            multiplier = 0.85
            reasoning_parts.append("Multi-jurisdiction route detected")
        
        # Code-share flight detection
        if any(indicator in airline for indicator in self.code_share_indicators):
            score *= 0.90
            multiplier = 0.90
            reasoning_parts.append("Code-share flight detected")
        
        # Missing route information
        if not origin or not destination:
            score *= 0.3
            reasoning_parts.append("Missing flight route information")
        
        # Jurisdiction-specific clarity checks
        if jurisdiction_result == "APPR":
            canadian_indicators = ['yyz', 'yvr', 'yul', 'yyc', 'yow', 'canada']
            if not any(indicator in origin or indicator in destination 
                      for indicator in canadian_indicators):
                score *= 0.6
                reasoning_parts.append("Route doesn't clearly indicate Canadian jurisdiction")
        
        elif jurisdiction_result == "EU261":
            eu_indicators = ['fra', 'cdg', 'mad', 'bcn', 'fco', 'ams', 'bru', 'germany', 'france', 'spain']
            if not any(indicator in origin or indicator in destination 
                      for indicator in eu_indicators):
                score *= 0.5
                reasoning_parts.append("Route doesn't clearly indicate EU jurisdiction")
        
        # Reasoning quality
        if len(reasoning) < 20:
            score *= 0.8
            reasoning_parts.append("Insufficient jurisdiction reasoning")
        
        return RiskFactor(
            name="Jurisdiction Clarity",
            weight=self.risk_weights['jurisdiction_clarity'],
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "High jurisdiction clarity",
            multiplier=multiplier
        )

    def _assess_legal_complexity(self, flight_data: Dict[str, Any], 
                               eligibility_result: Dict[str, Any], 
                               jurisdiction_result: str) -> RiskFactor:
        """Assess legal complexity including borderline cases"""
        score = 1.0
        reasoning_parts = []
        
        delay_length = flight_data.get('delay_length', 0)
        delay_reason = flight_data.get('delay_reason', '').lower()
        
        # Borderline delay durations (near threshold boundaries)
        if 2.5 <= delay_length <= 3.5:  # Near 3-hour threshold
            score *= 0.7
            reasoning_parts.append("Delay duration near compensation threshold")
        
        if 8.5 <= delay_length <= 9.5:  # Near 9-hour threshold
            score *= 0.7
            reasoning_parts.append("Delay duration near higher compensation threshold")
        
        # Extraordinary circumstances gray areas
        if any(ec in delay_reason for ec in self.extraordinary_circumstances):
            if 'severe' not in delay_reason and 'extreme' not in delay_reason:
                score *= 0.6
                reasoning_parts.append("Extraordinary circumstances require legal interpretation")
        
        # Multiple delay reasons or conflicting information
        if 'and' in delay_reason or 'also' in delay_reason:
            score *= 0.6
            reasoning_parts.append("Multiple delay reasons increase legal complexity")
        
        # Jurisdiction complexity
        if jurisdiction_result == "NEITHER":
            score *= 0.5
            reasoning_parts.append("No clear jurisdiction increases legal complexity")
        
        return RiskFactor(
            name="Legal Complexity",
            weight=self.risk_weights['legal_complexity'],
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Low legal complexity"
        )

    def _assess_delay_reason_ambiguity(self, flight_data: Dict[str, Any], 
                                     eligibility_result: Dict[str, Any]) -> RiskFactor:
        """Assess delay reason ambiguity and clarity"""
        score = 1.0
        reasoning_parts = []
        
        delay_reason = flight_data.get('delay_reason', '').lower()
        
        # Highly ambiguous reasons
        if any(ambiguous in delay_reason for ambiguous in self.ambiguous_delay_reasons):
            score *= 0.4
            reasoning_parts.append("Airline provided ambiguous delay reason")
        
        # Weather-related ambiguity
        if 'weather' in delay_reason:
            if not any(specific in delay_reason for specific in ['storm', 'severe', 'extreme', 'hurricane', 'blizzard']):
                score *= 0.7
                reasoning_parts.append("Weather-related delay lacks specificity")
        
        # Missing delay reason
        if not delay_reason or delay_reason.strip() == '':
            score *= 0.3
            reasoning_parts.append("No delay reason provided")
        
        # Vague time references
        if any(vague in delay_reason for vague in ['some time', 'a while', 'delayed', 'late']):
            score *= 0.6
            reasoning_parts.append("Vague delay reason provided")
        
        return RiskFactor(
            name="Delay Reason Ambiguity",
            weight=self.risk_weights['delay_reason_ambiguity'],
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Clear delay reason provided"
        )

    def _assess_data_completeness(self, flight_data: Dict[str, Any], 
                                eligibility_result: Dict[str, Any]) -> RiskFactor:
        """Assess completeness of available data"""
        score = 1.0
        reasoning_parts = []
        
        required_fields = ['flight_number', 'flight_date', 'airline', 'origin', 'destination', 'delay_length']
        missing_fields = []
        
        for field in required_fields:
            if not flight_data.get(field) or flight_data.get(field) == '':
                missing_fields.append(field)
        
        if missing_fields:
            score *= 0.8 ** len(missing_fields)  # Exponential penalty for missing data
            reasoning_parts.append(f"Missing data: {', '.join(missing_fields)}")
        
        # Check for passenger uncertainty indicators
        if 'passenger_notes' in flight_data:
            notes = flight_data['passenger_notes'].lower()
            if any(uncertain in notes for uncertain in self.uncertainty_indicators):
                score *= 0.8
                reasoning_parts.append("Passenger expressed uncertainty about details")
        
        # Legal citations availability
        legal_citations = eligibility_result.get('legal_citations', [])
        if not legal_citations or len(legal_citations) == 0:
            score *= 0.6
            reasoning_parts.append("No specific legal citations available")
        
        return RiskFactor(
            name="Data Completeness",
            weight=self.risk_weights['data_completeness'],
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Complete data available"
        )

    def _assess_regulatory_edge_cases(self, flight_data: Dict[str, Any], 
                                    jurisdiction_result: str, 
                                    eligibility_result: Dict[str, Any]) -> RiskFactor:
        """Assess regulatory edge cases and special circumstances"""
        score = 1.0
        reasoning_parts = []
        
        flight_date = flight_data.get('flight_date', '')
        delay_length = flight_data.get('delay_length', 0)
        
        # Holiday period detection (simplified)
        holiday_months = ['12', '01', '07', '08']  # Dec, Jan, Jul, Aug
        if any(month in flight_date for month in holiday_months):
            score *= 0.9
            reasoning_parts.append("Flight during holiday period")
        
        # Multi-airline scenarios
        airline = flight_data.get('airline', '').lower()
        if 'operated by' in airline or 'marketed by' in airline:
            score *= 0.8
            reasoning_parts.append("Multi-airline flight scenario")
        
        # Very long delays (potential extraordinary circumstances)
        if delay_length > 12:
            score *= 0.7
            reasoning_parts.append("Very long delay may involve extraordinary circumstances")
        
        # Weekend flights (different operational patterns)
        # This would need actual date parsing in a real implementation
        
        return RiskFactor(
            name="Regulatory Edge Cases",
            weight=self.risk_weights['regulatory_edge_cases'],
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Standard regulatory scenario"
        )

    def _assess_financial_impact(self, flight_data: Dict[str, Any], 
                               eligibility_result: Dict[str, Any]) -> RiskFactor:
        """Assess financial impact and compensation amount"""
        score = 1.0
        reasoning_parts = []
        
        compensation_amount = eligibility_result.get('compensation_amount', 0)
        delay_length = flight_data.get('delay_length', 0)
        
        # High-value claims need extra scrutiny
        if compensation_amount > 1000:
            score *= 0.8
            reasoning_parts.append("High-value claim requires additional scrutiny")
        
        # Multiple passengers
        passenger_count = flight_data.get('passenger_count', 1)
        if passenger_count > 4:
            score *= 0.9
            reasoning_parts.append("Multiple passengers increase financial impact")
        
        # Long delays with high compensation
        if delay_length > 6 and compensation_amount > 500:
            score *= 0.85
            reasoning_parts.append("Long delay with significant compensation")
        
        return RiskFactor(
            name="Financial Impact",
            weight=self.risk_weights['financial_impact'],
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Standard financial impact"
        )

    def _assess_precedent_similarity(self, flight_data: Dict[str, Any], 
                                   eligibility_result: Dict[str, Any]) -> RiskFactor:
        """Assess similarity to existing precedents"""
        score = 1.0
        reasoning_parts = []
        
        # This would typically involve vector similarity search against case database
        # For now, we'll use heuristics based on delay reason and jurisdiction
        
        delay_reason = flight_data.get('delay_reason', '').lower()
        jurisdiction = eligibility_result.get('jurisdiction', '')
        
        # Novel delay reasons
        novel_reasons = ['cyber attack', 'pilot shortage', 'fuel contamination', 'cargo issue']
        if any(novel in delay_reason for novel in novel_reasons):
            score *= 0.6
            reasoning_parts.append("Novel delay reason with limited precedents")
        
        # Complex jurisdiction scenarios
        if jurisdiction == "NEITHER":
            score *= 0.7
            reasoning_parts.append("No clear jurisdiction - limited precedents")
        
        return RiskFactor(
            name="Precedent Similarity",
            weight=self.risk_weights['precedent_similarity'],
            score=score,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Similar to existing precedents"
        )

    def _detect_patterns(self, flight_data: Dict[str, Any], 
                        eligibility_result: Dict[str, Any]) -> List[str]:
        """Detect complex patterns that require special handling"""
        patterns = []
        
        origin = flight_data.get('origin', '').lower()
        destination = flight_data.get('destination', '').lower()
        airline = flight_data.get('airline', '').lower()
        delay_reason = flight_data.get('delay_reason', '').lower()
        
        # Multi-jurisdiction routes
        if any(indicator in origin or indicator in destination 
               for indicator in self.multi_jurisdiction_indicators):
            patterns.append("Multi-jurisdiction route")
        
        # Code-share flights
        if any(indicator in airline for indicator in self.code_share_indicators):
            patterns.append("Code-share flight")
        
        # Borderline delay durations
        delay_length = flight_data.get('delay_length', 0)
        if 2.8 <= delay_length <= 3.2:
            patterns.append("Borderline 3-hour delay")
        elif 8.8 <= delay_length <= 9.2:
            patterns.append("Borderline 9-hour delay")
        
        # Extraordinary circumstances gray areas
        if any(ec in delay_reason for ec in self.extraordinary_circumstances):
            if 'severe' not in delay_reason and 'extreme' not in delay_reason:
                patterns.append("Extraordinary circumstances gray area")
        
        # Conflicting information
        if 'and' in delay_reason or 'also' in delay_reason:
            patterns.append("Multiple delay reasons")
        
        return patterns

    def _analyze_conversation_context(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Analyze conversation history for contextual factors"""
        contextual_factors = []
        
        # Look for uncertainty indicators in conversation
        for message in conversation_history:
            if message.get('message_type') == 'user':
                content = message.get('content', '').lower()
                if any(uncertain in content for uncertain in self.uncertainty_indicators):
                    contextual_factors.append("Passenger expressed uncertainty")
                    break
        
        # Look for multiple delay reasons mentioned
        delay_reasons_mentioned = 0
        for message in conversation_history:
            if message.get('message_type') == 'user':
                content = message.get('content', '')
                if 'delay' in content.lower() and 'reason' in content.lower():
                    delay_reasons_mentioned += 1
        
        if delay_reasons_mentioned > 1:
            contextual_factors.append("Multiple delay reasons mentioned in conversation")
        
        # Look for time-sensitive factors
        for message in conversation_history:
            if message.get('message_type') == 'user':
                content = message.get('content', '').lower()
                if any(urgent in content for urgent in ['urgent', 'asap', 'quickly', 'soon']):
                    contextual_factors.append("Time-sensitive request")
                    break
        
        return contextual_factors

    def _calculate_weighted_confidence(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate weighted overall confidence score"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor in risk_factors:
            # Apply any multipliers
            adjusted_score = factor.score * factor.multiplier
            weighted_sum += adjusted_score * factor.weight
            total_weight += factor.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_risk_level(self, confidence: float) -> RiskLevel:
        """Determine risk level based on confidence score"""
        if confidence >= self.risk_thresholds[RiskLevel.LOW]:
            return RiskLevel.LOW
        elif confidence >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        elif confidence >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _determine_handoff_requirements(self, risk_level: RiskLevel, 
                                      risk_factors: List[RiskFactor], 
                                      patterns_detected: List[str]) -> Tuple[bool, str]:
        """Determine handoff requirements based on risk level and factors"""
        if risk_level == RiskLevel.LOW:
            return False, "Auto-process with high confidence"
        elif risk_level == RiskLevel.MEDIUM:
            return True, "Review within 24 hours"
        elif risk_level == RiskLevel.HIGH:
            return True, "Priority review within 1 hour"
        else:  # CRITICAL
            return True, "Immediate human intervention"

    def _generate_reasoning(self, risk_factors: List[RiskFactor], 
                          patterns_detected: List[str], 
                          contextual_factors: List[str]) -> str:
        """Generate comprehensive reasoning for the risk assessment"""
        reasoning_parts = []
        
        # Overall confidence
        overall_confidence = self._calculate_weighted_confidence(risk_factors)
        reasoning_parts.append(f"Overall confidence: {overall_confidence:.2f}")
        
        # Key risk factors
        high_risk_factors = [f for f in risk_factors if f.score < 0.7]
        if high_risk_factors:
            factor_names = [f.name for f in high_risk_factors]
            reasoning_parts.append(f"Key risk factors: {', '.join(factor_names)}")
        
        # Patterns detected
        if patterns_detected:
            reasoning_parts.append(f"Patterns detected: {', '.join(patterns_detected)}")
        
        # Contextual factors
        if contextual_factors:
            reasoning_parts.append(f"Contextual factors: {', '.join(contextual_factors)}")
        
        return ". ".join(reasoning_parts) + "."
