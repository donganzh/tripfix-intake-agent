# TripFix AI Intake Chatbot

An intelligent flight delay compensation intake system using LangChain, LangGraph, and Streamlit.

## Architecture Overview

### Core Components

1. **Vector Database (ChromaDB)**: Stores processed regulation documents (APPR & EU261) for semantic search
2. **LangGraph Workflow**: Orchestrates the multi-step intake process with human-in-the-loop logic
3. **Confidence Scoring**: Evaluates certainty of jurisdiction and eligibility decisions
4. **SQLite Database**: Persists intake sessions and conversation history
5. **Streamlit Interface**: Modern chat-based user interface

### Workflow Design

```
User Message → Extract Flight Info → Validate Data → Determine Jurisdiction → 
Assess Eligibility → Score Confidence → [High Confidence: Complete] or [Low Confidence: Human Handoff]
```

### Key Features

- **Empathetic Communication**: LLM-generated dynamic questions with emotional intelligence
- **Dual-Jurisdiction Support**: Handles both Canadian APPR and EU Regulation 261/2004
- **Confidence-Based HITL**: Automatically flags complex cases for human review
- **Robust PDF Processing**: Extracts and chunks regulation documents for vector search
- **Real-time Agent Status**: Shows users what the AI is working on behind the scenes

## Setup Instructions

### 1. Environment Setup
```bash
# Clone and navigate to project
cd tripfix_intake

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Data Setup
Place your regulation PDF files in the `data/regulations/` folder:
- Canadian Air Passenger Rights (APPR) documents
- EU Regulation 261/2004 documents

### 4. Initialize Database
```bash
# The system will automatically initialize the database and vector store on first run
python -c "from utils.database import IntakeDatabase; IntakeDatabase()"
```

### 5. Run the Application
```bash
streamlit run app.py
```

## Testing

Run the test scenarios:
```bash
pytest test_scenarios.py -v
```

### Test Coverage
- ✅ Canadian domestic flight (APPR jurisdiction)
- ✅ EU to Canada flight (EU261 jurisdiction)  
- ✅ US domestic flight (no jurisdiction)
- ✅ Ambiguous delay reason (human handoff)
- ✅ Complete intake process
- ✅ Off-topic conversation handling

## Database Schema

### intake_sessions
- `id`: Unique session identifier
- `flight_data`: JSON with flight details
- `jurisdiction`: APPR/EU261/NEITHER
- `jurisdiction_confidence`: Confidence score (0-1)
- `eligibility_result`: JSON with eligibility determination
- `eligibility_confidence`: Confidence score (0-1)
- `handoff_reason`: Why human review was triggered
- `completed`: Whether intake is finished

### conversation_history
- `session_id`: Links to intake_sessions
- `message_type`: user/assistant
- `content`: Message content and metadata
- `timestamp`: When message was sent

## Confidence Scoring Logic

### Jurisdiction Confidence Factors:
- Route clarity (clear EU/Canadian airports vs ambiguous)
- Airline country alignment with route
- Multiple jurisdiction applicability
- Reasoning quality and detail

### Eligibility Confidence Factors:
- Delay reason ambiguity (e.g., "operational reasons")
- Weather edge cases
- Borderline delay durations
- Legal citation availability

### Handoff Thresholds:
- Jurisdiction: < 0.70 confidence
- Eligibility: < 0.75 confidence

## Key Design Decisions

### 1. ChromaDB for Vector Storage
- **Why**: Runs locally, excellent for legal document retrieval
- **Benefits**: No external dependencies, good performance, easy setup
- **Alternative considered**: Pinecone (rejected for local requirement)

### 2. LangGraph for Orchestration
- **Why**: Explicit workflow control with conditional branching
- **Benefits**: Debuggable, stateful, handles complex decision trees
- **Alternative considered**: Simple LangChain chains (too linear)

### 3. Dual-Agent Architecture
- **Jurisdiction Agent**: Specializes in determining applicable laws
- **Eligibility Agent**: Focuses on compensation assessment
- **Benefits**: Separation of concerns, specialized prompting, better accuracy

### 4. SQLite for Session Storage
- **Why**: Local, reliable, good for structured intake data
- **Benefits**: ACID compliance, familiar SQL interface, no setup overhead

## API Usage

The system uses OpenAI API for:
- **GPT-4 Turbo**: Main conversation and legal analysis
- **text-embedding-3-small**: Document embeddings for vector search

Estimated costs:
- Per intake session: ~$0.10-0.50 depending on complexity
- Initial setup (PDF processing): ~$2-5 for typical regulation documents

## Deployment Considerations

### Production Readiness Checklist:
- [ ] Add rate limiting for API calls
- [ ] Implement proper error handling and logging
- [ ] Add user authentication
- [ ] Set up monitoring and analytics
- [ ] Configure production database (PostgreSQL)
- [ ] Add data backup and recovery
- [ ] Implement security headers and CORS
- [ ] Add comprehensive test coverage

### Scaling Considerations:
- Vector store can handle 100k+ documents
- SQLite suitable for <1000 concurrent users
- LangGraph workflow is stateless and horizontally scalable
- Consider Redis for session storage at scale

## Troubleshooting

### Common Issues:
1. **OpenAI API errors**: Check API key validity and rate limits
2. **PDF processing fails**: Ensure PDFs are text-based, not scanned images
3. **Vector store initialization slow**: Normal for first run with large PDFs
4. **Session not persisting**: Check database file permissions

### Debug Mode:
Set `DEBUG=true` in environment to enable verbose logging.

## Future Enhancements

### Phase 2 Features:
- Multi-language support
- Integration with airline APIs for real-time flight data
- Automated claim filing
- SMS/email notifications
- Advanced analytics dashboard

### Technical Improvements:
- Streaming responses for better UX
- Fine-tuned models for legal domain
- Advanced confidence scoring with ML models
- Integration with legal case management systems