from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from typing import Dict, Any, Tuple, List
import json
import logging

# Configure logging for agents
logger = logging.getLogger(__name__)

class JurisdictionAgent:
    def __init__(self, openai_api_key: str, vector_store):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=openai_api_key,
            temperature=0.1
        )
        self.vector_store = vector_store
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal expert specializing in air passenger rights legislation. 
            
Your task is to determine which jurisdiction's laws apply to a flight delay compensation claim:
1. Canadian Air Passenger Rights (APPR) 
2. EU Regulation 261/2004
3. Neither (no applicable jurisdiction)

Key rules:
- APPR applies to flights within/from/to Canada on Canadian airlines, or domestic Canadian flights
- EU 261 applies to flights departing from EU, or arriving in EU on EU airlines
- Consider the airline's country of origin and flight route carefully
- If multiple jurisdictions could apply, choose the most favorable to the passenger

Flight details:
{flight_data}

Relevant regulation excerpts:
{relevant_regulations}

Respond in JSON format:
{{
    "jurisdiction": "APPR|EU261|NEITHER",
    "reasoning": "detailed explanation of your decision",
    "applicable_articles": ["list of specific regulation sections that apply"]
}}"""),
            ("human", "Please analyze this flight and determine the applicable jurisdiction.")
        ])
    
    def determine_jurisdiction(self, flight_data: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """Determine which jurisdiction applies to the flight"""
        logger.info(f"🌍 JurisdictionAgent: Starting jurisdiction determination")
        
        # Search for relevant regulations
        search_query = f"jurisdiction rules {flight_data.get('origin', '')} to {flight_data.get('destination', '')} {flight_data.get('airline', '')}"
        logger.info(f"🔍 Searching regulations with query: {search_query}")
        relevant_docs = self.vector_store.search(search_query, n_results=8)
        logger.info(f"📚 Found {len(relevant_docs)} relevant regulation documents")
        
        regulations_text = "\n\n".join([f"Source: {doc['metadata']['source']}\n{doc['content']}" 
                                      for doc in relevant_docs])
        
        # Format flight data for prompt
        flight_summary = json.dumps(flight_data, indent=2)
        logger.info("🧠 Calling LLM for jurisdiction determination...")
        
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({
                "flight_data": flight_summary,
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
            
            # Validate required fields
            jurisdiction = result.get("jurisdiction", "NEITHER")
            if jurisdiction not in ["APPR", "EU261", "NEITHER"]:
                jurisdiction = "NEITHER"
            
            logger.info(f"✅ JurisdictionAgent: Determination complete - Jurisdiction: {jurisdiction}")
            
            return (
                jurisdiction,
                result.get("reasoning", "No reasoning provided"),
                result.get("applicable_articles", [])
            )
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in jurisdiction determination: {e}")
            print(f"Response content: {response.content if 'response' in locals() else 'No response'}")
            return "NEITHER", f"JSON parsing error: {str(e)}", []
        except Exception as e:
            print(f"Error in jurisdiction determination: {e}")
            return "NEITHER", f"Error in analysis: {str(e)}", []