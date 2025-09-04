# TripFix Intake Agent Prompt Engineering Documentation

## Overview

This document provides a comprehensive overview of the prompt engineering approach used in the TripFix intake agent workflow. The system uses a multi-agent architecture with specialized prompts for different stages of the customer interaction and legal analysis process.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Prompt Engineering Principles](#core-prompt-engineering-principles)
3. [Agent-Specific Prompts](#agent-specific-prompts)
4. [Dynamic Prompt Generation](#dynamic-prompt-generation)
5. [Prompt Templates and Variations](#prompt-templates-and-variations)
6. [Error Handling and Fallbacks](#error-handling-and-fallbacks)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices and Lessons Learned](#best-practices-and-lessons-learned)

## Architecture Overview

The TripFix system uses a LangGraph-based workflow with specialized agents:

- **IntakeAgent**: Main orchestrator with conversational prompts
- **JurisdictionAgent**: Legal analysis for determining applicable regulations
- **EligibilityAgent**: Compensation eligibility assessment
- **ConfidenceScorer**: Risk assessment and confidence scoring
- **AdvancedConfidenceEngine**: Multi-factor risk analysis

## Core Prompt Engineering Principles

### 1. **Human-Like Conversation**
- Avoid robotic, template-like responses
- Use varied language and personality styles
- Implement contextual awareness and memory
- Maintain conversational flow and natural transitions

### 2. **Legal Accuracy and Precision**
- Structured JSON output for legal analysis
- Specific regulation citations and references
- Clear reasoning chains for legal decisions
- Confidence scoring for risk assessment

### 3. **Dynamic Adaptation**
- Context-aware prompt generation
- Style variation based on user mood and situation
- Adaptive questioning based on collected information
- Personalized responses using user data

### 4. **Error Resilience**
- Fallback prompts for failed generations
- JSON parsing error handling
- Graceful degradation for edge cases
- Multiple retry mechanisms

## Agent-Specific Prompts

### IntakeAgent - Conversational Prompts

#### 1. **Initial Greeting Prompt**

```python
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
```

**Key Features:**
- Dynamic style selection (enthusiastic, empathetic, professional, caring)
- Personality injection to avoid robotic responses
- Contextual awareness of customer situation
- Natural conversation flow

#### 2. **Information Collection Prompts**

```python
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
```

**Key Features:**
- Context-aware questioning based on previously collected data
- Multiple conversation styles (empathetic, professional, friendly, etc.)
- Dynamic field-specific guidance
- Natural conversation flow maintenance

#### 3. **Document Collection Prompt**

```python
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
```

### JurisdictionAgent - Legal Analysis Prompts

#### **Jurisdiction Determination Prompt**

```python
jurisdiction_prompt = ChatPromptTemplate.from_messages([
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
{
    "jurisdiction": "APPR|EU261|NEITHER",
    "reasoning": "detailed explanation of your decision",
    "applicable_articles": ["list of specific regulation sections that apply"]
}"""),
    ("human", "Please analyze this flight and determine the applicable jurisdiction.")
])
```

**Key Features:**
- Structured legal analysis with clear decision criteria
- JSON output format for programmatic processing
- Integration with vector store for relevant regulations
- Specific regulation citations and reasoning

### EligibilityAgent - Compensation Analysis Prompts

#### **Eligibility Assessment Prompt**

```python
eligibility_prompt = ChatPromptTemplate.from_messages([
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
{
    "eligible": true/false,
    "compensation_amount": number_or_null,
    "reasoning": "detailed legal reasoning",
    "legal_citations": ["specific regulation sections cited"],
    "delay_category": "controllable|uncontrollable|extraordinary_circumstances",
    "key_factors": ["list of decisive factors in determination"]
}"""),
    ("human", "Please determine eligibility for compensation based on the provided information.")
])
```

**Key Features:**
- Comprehensive legal analysis framework
- Structured output with compensation amounts
- Legal citation tracking
- Delay categorization for legal clarity

## Dynamic Prompt Generation

### 1. **Style Variation System**

The system uses a dynamic style selection mechanism to avoid repetitive responses:

```python
conversation_styles = [
    "empathetic_helper",
    "professional_advisor", 
    "friendly_neighbor",
    "understanding_friend",
    "efficient_specialist",
    "caring_supporter"
]

style_instructions = {
    "empathetic_helper": "Be deeply empathetic and understanding. Show genuine care for their situation. Use phrases like 'I can only imagine how frustrating this must be' or 'That sounds really difficult'.",
    "professional_advisor": "Be professional but warm. Show expertise and confidence. Use phrases like 'Let me help you with that' or 'I'll make sure we get this sorted'.",
    "friendly_neighbor": "Be casual and friendly, like talking to a neighbor. Use phrases like 'Oh no, that's terrible!' or 'I'm so sorry that happened to you'.",
    "understanding_friend": "Be like a supportive friend who really gets it. Use phrases like 'I totally understand' or 'That's so unfair'.",
    "efficient_specialist": "Be efficient and focused on helping. Use phrases like 'Let's get this resolved quickly' or 'I'll make this as smooth as possible'.",
    "caring_supporter": "Be caring and supportive. Use phrases like 'I'm here to help you through this' or 'We'll get you the compensation you deserve'."
}
```

### 2. **Context-Aware Prompting**

Prompts adapt based on:
- User's name and mood
- Previously collected information
- Current conversation stage
- User's response patterns

### 3. **Temperature Control**

Different temperature settings for different use cases:
- **Creative responses**: `temperature=0.8` for varied, human-like conversations
- **Legal analysis**: `temperature=0.1` for consistent, accurate legal reasoning
- **Standard responses**: `temperature=0.3` for balanced creativity and consistency

## Prompt Templates and Variations

### 1. **Greeting Variations**

The system includes multiple greeting styles:

```python
greeting_styles = [
    "warm_empathetic",
    "professional_confident", 
    "friendly_casual",
    "caring_supportive"
]

greeting_style_instructions = {
    "warm_empathetic": "Be warm and empathetic. Show genuine understanding of their frustration. Use phrases like 'I'm so sorry you had to go through this' or 'That must have been incredibly frustrating'.",
    "professional_confident": "Be professional and confident. Show expertise and competence. Use phrases like 'I'm here to help you get the compensation you deserve' or 'Let's get this sorted for you'.",
    "friendly_casual": "Be friendly and casual. Sound like a helpful friend. Use phrases like 'Hey there!' or 'Oh no, that's terrible!'",
    "caring_supportive": "Be caring and supportive. Show genuine concern. Use phrases like 'I'm here to help you through this' or 'We'll make sure you get what you're owed'."
}
```

### 2. **Transition Prompts**

Smooth transitions between conversation stages:

```python
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
```

## Error Handling and Fallbacks

### 1. **JSON Parsing Error Handling**

```python
try:
    result = json.loads(response_text)
except json.JSONDecodeError as e:
    print(f"JSON parsing error in jurisdiction determination: {e}")
    print(f"Response content: {response.content if 'response' in locals() else 'No response'}")
    return "NEITHER", f"JSON parsing error: {str(e)}", []
```

### 2. **Fallback Prompts**

When LLM generation fails, the system uses fallback responses:

```python
try:
    chain = question_prompt | creative_llm
    response = chain.invoke({...})
    question = response.content
except Exception as e:
    print(f"Error generating question: {e}")
    # Fallback to simple question
    question = f"Could you please provide your {next_field.replace('_', ' ')}?"
```

### 3. **Response Validation**

The system validates LLM responses and provides fallbacks:

```python
# Validate required fields
jurisdiction = result.get("jurisdiction", "NEITHER")
if jurisdiction not in ["APPR", "EU261", "NEITHER"]:
    jurisdiction = "NEITHER"
```

## Performance Optimization

### 1. **Caching and Reuse**

- Prompt templates are cached and reused
- Vector store results are cached for similar queries
- Database connections are pooled

### 2. **Parallel Processing**

- Multiple agents can work simultaneously
- Vector searches are optimized for speed
- Database operations are batched

### 3. **Response Time Optimization**

- Streamlined prompt templates
- Efficient JSON parsing
- Minimal context switching

## Best Practices and Lessons Learned

### 1. **Prompt Engineering Best Practices**

#### **Do's:**
- Use specific, clear instructions
- Provide examples and context
- Structure output formats clearly
- Include error handling instructions
- Use temperature settings appropriately
- Implement style variation for naturalness

#### **Don'ts:**
- Avoid overly complex prompts
- Don't rely on single-shot generation
- Avoid hardcoded responses
- Don't ignore error cases
- Avoid repetitive language patterns

### 2. **Legal Analysis Specific Practices**

- Always provide legal reasoning
- Include specific regulation citations
- Use structured JSON output
- Implement confidence scoring
- Handle edge cases explicitly

### 3. **Conversational AI Practices**

- Maintain conversation context
- Use varied language patterns
- Implement personality consistency
- Handle user mood and emotions
- Provide natural transitions

### 4. **System Integration Practices**

- Use consistent data formats
- Implement proper error handling
- Maintain state across interactions
- Provide fallback mechanisms
- Log all interactions for debugging

## Advanced Features

### 1. **Multi-Factor Risk Assessment**

The Advanced Confidence Engine uses sophisticated prompts for risk analysis:

```python
def _assess_jurisdiction_clarity(self, flight_data: Dict[str, Any], 
                               jurisdiction_result: str, reasoning: str) -> RiskFactor:
    """Assess jurisdiction clarity with multi-jurisdiction detection"""
    # Complex risk factor analysis with weighted scoring
    # Pattern detection for edge cases
    # Confidence scoring based on multiple factors
```

### 2. **Pattern Detection**

The system detects complex patterns that require special handling:

```python
def _detect_patterns(self, flight_data: Dict[str, Any], 
                    eligibility_result: Dict[str, Any]) -> List[str]:
    """Detect complex patterns that require special handling"""
    patterns = []
    
    # Multi-jurisdiction routes
    # Code-share flights
    # Borderline delay durations
    # Extraordinary circumstances gray areas
    # Conflicting information
```

### 3. **Contextual Analysis**

The system analyzes conversation history for contextual factors:

```python
def _analyze_conversation_context(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
    """Analyze conversation history for contextual factors"""
    # Uncertainty indicators
    # Multiple delay reasons
    # Time-sensitive factors
    # Emotional context
```

## Conclusion

The TripFix prompt engineering approach combines:

1. **Human-like conversation** with dynamic style variation
2. **Legal precision** with structured analysis and citations
3. **Error resilience** with comprehensive fallback mechanisms
4. **Performance optimization** with caching and parallel processing
5. **Advanced risk assessment** with multi-factor analysis

This approach enables the system to provide both natural, empathetic customer service and accurate, reliable legal analysis for flight delay compensation claims.

## Future Improvements

1. **Enhanced Context Awareness**: Deeper understanding of user emotions and needs
2. **Advanced Legal Reasoning**: More sophisticated legal analysis capabilities
3. **Multi-language Support**: Internationalization of prompts and responses
4. **Learning from Feedback**: Continuous improvement based on user interactions
5. **Integration with External APIs**: Real-time flight data and regulation updates
