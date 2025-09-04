from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any, Tuple, List
import json
import logging

# Configure logging for agents
logger = logging.getLogger(__name__)

class EligibilityAgent:
    def __init__(self, openai_api_key: str, vector_store):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=openai_api_key,
            temperature=0.1
        )
        self.vector_store = vector_store
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal expert specializing in air passenger compensation claims.

Your task is to determine if a passenger is eligible for compensation under the specified jurisdiction's laws.

Flight Information:
{flight_data}

Applicable Jurisdiction: {jurisdiction}

Relevant Legal Text:
{relevant_regulations}

Consider these factors:
1. Delay duration thresholds
2. Airline responsibility (controllable vs uncontrollable delays)
3. Advance notice requirements
4. Extraordinary circumstances exceptions

Respond in JSON format:
{{
    "eligible": true/false,
    "compensation_amount": number_or_null,
    "reasoning": "detailed legal reasoning",
    "legal_citations": ["specific regulation sections cited"],
    "delay_category": "controllable|uncontrollable|extraordinary_circumstances",
    "key_factors": ["list of decisive factors in determination"]
}}"""),
            ("human", "Please determine eligibility for compensation based on the provided information.")
        ])
    
    def assess_eligibility(self, 
                          flight_data: Dict[str, Any], 
                          jurisdiction: str) -> Tuple[bool, float, str, List[str]]:
        """Assess eligibility for compensation"""
        logger.info(f"⚖️ EligibilityAgent: Starting eligibility assessment for {jurisdiction} jurisdiction")
        
        # Search for jurisdiction-specific regulations
        search_query = f"{jurisdiction} compensation eligibility delay {flight_data.get('delay_reason', '')} {flight_data.get('delay_length', 0)} hours"
        logger.info(f"🔍 Searching regulations with query: {search_query}")
        
        filter_metadata = {"regulation_type": jurisdiction} if jurisdiction in ["APPR", "EU261"] else None
        relevant_docs = self.vector_store.search(search_query, n_results=10, filter_metadata=filter_metadata)
        logger.info(f"📚 Found {len(relevant_docs)} relevant regulation documents")
        
        regulations_text = "\n\n".join([
            f"Source: {doc['metadata']['source']} (Regulation: {doc['metadata']['regulation_type']})\n{doc['content']}" 
            for doc in relevant_docs
        ])
        
        flight_summary = json.dumps(flight_data, indent=2)
        logger.info("🧠 Calling LLM for eligibility determination...")
        
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({
                "flight_data": flight_summary,
                "jurisdiction": jurisdiction,
                "relevant_regulations": regulations_text
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
            
            # Parse JSON response
            result = json.loads(response_text)
            
            # Validate and extract results
            eligible = result.get("eligible", False)
            compensation_amount = result.get("compensation_amount", 0.0)
            
            # Ensure compensation_amount is a number
            try:
                compensation_amount = float(compensation_amount)
            except (ValueError, TypeError):
                compensation_amount = 0.0
            
            logger.info(f"✅ EligibilityAgent: Assessment complete - Eligible: {eligible}, Compensation: ${compensation_amount}")
            
            return (
                bool(eligible),
                compensation_amount,
                result.get("reasoning", "No reasoning provided"),
                result.get("legal_citations", [])
            )
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in eligibility assessment: {e}")
            print(f"Response content: {response.content if 'response' in locals() else 'No response'}")
            return False, 0.0, f"JSON parsing error: {str(e)}", []
        except Exception as e:
            print(f"Error in eligibility assessment: {e}")
            return False, 0.0, f"Error in analysis: {str(e)}", []