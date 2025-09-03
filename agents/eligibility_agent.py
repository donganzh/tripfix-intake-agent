from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any, Tuple, List
import json

class EligibilityAgent:
    def __init__(self, openai_api_key: str, vector_store):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
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
        
        # Search for jurisdiction-specific regulations
        search_query = f"{jurisdiction} compensation eligibility delay {flight_data.get('delay_reason', '')} {flight_data.get('delay_length', 0)} hours"
        
        filter_metadata = {"regulation_type": jurisdiction} if jurisdiction in ["APPR", "EU261"] else None
        relevant_docs = self.vector_store.search(search_query, n_results=10, filter_metadata=filter_metadata)
        
        regulations_text = "\n\n".join([
            f"Source: {doc['metadata']['source']} (Regulation: {doc['metadata']['regulation_type']})\n{doc['content']}" 
            for doc in relevant_docs
        ])
        
        flight_summary = json.dumps(flight_data, indent=2)
        
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({
                "flight_data": flight_summary,
                "jurisdiction": jurisdiction,
                "relevant_regulations": regulations_text
            })
            
            result = json.loads(response.content)
            
            return (
                result.get("eligible", False),
                result.get("compensation_amount", 0.0),
                result.get("reasoning", "No reasoning provided"),
                result.get("legal_citations", [])
            )
        
        except Exception as e:
            print(f"Error in eligibility assessment: {e}")
            return False, 0.0, f"Error in analysis: {str(e)}", []