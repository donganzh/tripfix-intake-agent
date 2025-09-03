from typing import Dict, Any, Tuple
import re

class ConfidenceScorer:
    def __init__(self):
        self.jurisdiction_threshold = 0.7
        self.eligibility_threshold = 0.75
    
    def score_jurisdiction_confidence(self, 
                                   flight_data: Dict[str, Any], 
                                   jurisdiction_result: str,
                                   reasoning: str) -> Tuple[float, str]:
        """Score confidence in jurisdiction determination"""
        confidence = 1.0
        reasons = []
        
        # Check flight route clarity
        origin = flight_data.get('origin', '').strip()
        destination = flight_data.get('destination', '').strip()
        airline = flight_data.get('airline', '').strip()
        
        if not origin or not destination:
            confidence *= 0.3
            reasons.append("Missing flight route information")
        
        # Route complexity scoring
        if jurisdiction_result == "APPR":
            # Canadian flights should be clear
            canadian_airports = ['YYZ', 'YVR', 'YUL', 'YYC', 'YOW']
            if not any(code in origin.upper() for code in canadian_airports) and \
               not any(code in destination.upper() for code in canadian_airports):
                if 'canada' not in origin.lower() and 'canada' not in destination.lower():
                    confidence *= 0.6
                    reasons.append("Route doesn't clearly indicate Canadian jurisdiction")
        
        elif jurisdiction_result == "EU261":
            # EU flights complexity
            eu_countries = ['germany', 'france', 'spain', 'italy', 'netherlands', 'belgium']
            eu_codes = ['FRA', 'CDG', 'MAD', 'BCN', 'FCO', 'AMS', 'BRU']
            
            eu_origin = any(country in origin.lower() for country in eu_countries) or \
                       any(code in origin.upper() for code in eu_codes)
            eu_dest = any(country in destination.lower() for country in eu_countries) or \
                     any(code in destination.upper() for code in eu_codes)
            
            if not eu_origin and not eu_dest:
                confidence *= 0.5
                reasons.append("Route doesn't clearly indicate EU jurisdiction")
        
        # Check reasoning quality
        if len(reasoning) < 50:
            confidence *= 0.8
            reasons.append("Insufficient reasoning provided")
        
        # Ambiguous airline cases
        if 'unclear' in reasoning.lower() or 'ambiguous' in reasoning.lower():
            confidence *= 0.6
            reasons.append("Ambiguous route or airline information")
        
        explanation = f"Jurisdiction confidence: {confidence:.2f}. " + "; ".join(reasons) if reasons else "High confidence in jurisdiction determination"
        
        return confidence, explanation
    
    def score_eligibility_confidence(self, 
                                   eligibility_data: Dict[str, Any],
                                   legal_citations: List[str]) -> Tuple[float, str]:
        """Score confidence in eligibility determination"""
        confidence = 1.0
        reasons = []
        
        delay_length = eligibility_data.get('delay_length', 0)
        delay_reason = eligibility_data.get('delay_reason', '').lower()
        
        # Ambiguous delay reasons significantly reduce confidence
        ambiguous_reasons = [
            'operational reasons', 'technical issues', 'crew scheduling',
            'operational requirements', 'network optimization', 'unforeseen circumstances'
        ]
        
        if any(ambiguous in delay_reason for ambiguous in ambiguous_reasons):
            confidence *= 0.4
            reasons.append("Airline provided ambiguous delay reason requiring legal interpretation")
        
        # Weather-related edge cases
        if 'weather' in delay_reason:
            if 'storm' not in delay_reason and 'severe' not in delay_reason:
                confidence *= 0.7
                reasons.append("Weather-related delay may need specific assessment")
        
        # Missing delay length
        if delay_length == 0:
            confidence *= 0.5
            reasons.append("Missing or unclear delay duration")
        
        # Citation quality
        if not legal_citations or len(legal_citations) == 0:
            confidence *= 0.6
            reasons.append("No specific legal citations found")
        
        # Borderline compensation amounts
        if delay_length > 0 and delay_length < 4:  # 3-4 hour edge case
            confidence *= 0.8
            reasons.append("Delay duration in borderline compensation range")
        
        explanation = f"Eligibility confidence: {confidence:.2f}. " + "; ".join(reasons) if reasons else "High confidence in eligibility determination"
        
        return confidence, explanation
    
    def should_handoff_to_human(self, jurisdiction_confidence: float, eligibility_confidence: float) -> Tuple[bool, str]:
        """Determine if case should be handed off to human"""
        if jurisdiction_confidence < self.jurisdiction_threshold:
            return True, f"Jurisdiction determination confidence too low ({jurisdiction_confidence:.2f} < {self.jurisdiction_threshold})"
        
        if eligibility_confidence < self.eligibility_threshold:
            return True, f"Eligibility determination confidence too low ({eligibility_confidence:.2f} < {self.eligibility_threshold})"
        
        return False, "Confidence levels acceptable for automated processing"