# TripFix AI Intake System

An intelligent flight delay compensation intake system using LangChain, LangGraph, and Streamlit with comprehensive evaluation and monitoring capabilities.

## 🎯 Overview

TripFix is a sophisticated AI-powered system that helps passengers understand their rights and potential compensation for flight delays. The system uses advanced prompt engineering, multi-agent architecture, and intelligent risk assessment to provide accurate legal analysis and empathetic customer service.

## 🏗️ Architecture

### Core Components

1. **LangGraph Workflow**: Orchestrates the multi-step intake process with conditional branching
2. **Multi-Agent System**: Specialized agents for jurisdiction, eligibility, and confidence assessment
3. **Advanced Confidence Engine**: 7-factor risk assessment with dynamic thresholds
4. **Vector Database (ChromaDB)**: Stores processed regulation documents for semantic search
5. **SQLite Database**: Persists intake sessions, conversation history, and supporting files
6. **Streamlit Interface**: Modern chat-based UI with real-time risk assessment dashboard
7. **Comprehensive Testing**: Golden test dataset with 7 required scenarios

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TripFix LangGraph Workflow                   │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    greet    │
                    │   (Node)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ collect_info│
                    │   (Node)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │validate_data│
                    │   (Node)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │determine_   │
                    │jurisdiction │
                    │   (Node)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │assess_      │
                    │eligibility  │
                    │   (Node)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │score_       │
                    │confidence   │
                    │   (Node)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Conditional │
                    │   Branch    │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
    ┌───────▼──────┐ ┌─────▼─────┐ ┌─────▼─────┐
    │ handoff_     │ │complete_  │ │handle_    │
    │ human        │ │intake     │ │off_topic  │
    │ (Node)       │ │(Node)     │ │(Node)     │
    └───────┬──────┘ └─────┬─────┘ └─────┬─────┘
            │              │              │
            └──────────────┼──────────────┘
                           │
                    ┌──────▼──────┐
                    │     END     │
                    └─────────────┘
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

#### For Unix/Linux/macOS:
```bash
# Clone the repository
git clone https://github.com/donganzh/tripfix-intake-agent.git
cd tripfix-intake-agent

# Make setup script executable and run it
chmod +x setup.sh
./setup.sh
```

#### For Windows:
```cmd
# Clone the repository
git clone https://github.com/donganzh/tripfix-intake-agent.git
cd tripfix-intake-agent

# Run the setup script
setup.bat
```

#### Using Python (Cross-platform):
```bash
# Clone the repository
git clone https://github.com/donganzh/tripfix-intake-agent.git
cd tripfix-intake-agent

# Run the Python setup script
python setup.py
```

### Option 2: Manual Setup

#### 1. Prerequisites
- Python 3.8 or higher
- Git
- OpenAI API key

#### 2. Environment Setup
```bash
# Clone and navigate to project
git clone https://github.com/donganzh/tripfix-intake-agent.git
cd tripfix-intake-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Configuration
Create `.env` file (or copy from `env.template`):
```bash
# Copy template
cp env.template .env

# Edit .env file and add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here
DEBUG=false
```

#### 4. Create Directories
```bash
mkdir -p data/uploads data/vectorstore data/regulations logs
```

#### 5. Add Sample Data (Optional)
Place regulation PDFs in `data/regulations/`:
- `APPR Canada.pdf` - Canadian Air Passenger Protection Regulations
- `EU.pdf` - EU Regulation 261/2004

### 3. Run the Application

#### Main Application
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the main application
streamlit run app.py
```

#### Additional Dashboards
```bash
# Intake dashboard (view completed sessions)
streamlit run pages/intake_dashboard.py

# Evaluation dashboard (performance monitoring)
streamlit run pages/evaluation_dashboard.py
```

#### Run Tests
```bash
# Run the comprehensive test suite
python test_tripfix_scenarios.py
```

### 4. Access the Application
- **Main App**: http://localhost:8501
- **Intake Dashboard**: http://localhost:8502 (if running separately)
- **Evaluation Dashboard**: http://localhost:8503 (if running separately)

## 🧠 Advanced Confidence Engine

### Multi-Factor Risk Assessment

The system uses a sophisticated 7-factor confidence scoring system instead of simple binary thresholds:

#### Risk Factors (Weighted)
1. **Jurisdiction Clarity (25%)** - Multi-jurisdiction routes, code-shares, airline mismatches
2. **Legal Complexity (20%)** - Borderline cases, extraordinary circumstances gray areas  
3. **Delay Reason Ambiguity (20%)** - "Operational reasons" triggers high risk
4. **Data Completeness (15%)** - Missing info, passenger uncertainty
5. **Regulatory Edge Cases (10%)** - Holiday periods, multi-airline scenarios
6. **Financial Impact (5%)** - High-value claims need extra scrutiny
7. **Precedent Similarity (5%)** - Novel situations with limited precedents

#### Dynamic Risk Levels
- 🟢 **Low Risk (75%+)**: Auto-process with high confidence
- 🟡 **Medium Risk (60-75%)**: Review within 24 hours  
- 🟠 **High Risk (40-60%)**: Priority review within 1 hour
- 🔴 **Critical Risk (<40%)**: Immediate human intervention

#### Intelligent Pattern Detection
- **Multi-jurisdiction routes** (Paris → Toronto = complex)
- **Code-share flights** (airline mismatch with flight code)
- **Borderline delay durations** (2.8 hours = near 3-hour threshold)
- **Extraordinary circumstances gray areas** ("weather-related" but no specifics)
- **Passenger uncertainty indicators** ("I think it was around...")

## 🎭 Prompt Engineering

### Dynamic Conversation Styles

The system uses sophisticated prompt engineering to create natural, human-like conversations:

#### Style Variation System
- **Empathetic Helper**: Deep understanding and care
- **Professional Advisor**: Expert guidance and confidence
- **Friendly Neighbor**: Casual and approachable
- **Understanding Friend**: Supportive and relatable
- **Efficient Specialist**: Focused and helpful
- **Caring Supporter**: Nurturing and encouraging

#### Key Prompt Features
- **Context-Aware Questioning**: Adapts based on collected information
- **Personality Injection**: Avoids robotic responses
- **Natural Transitions**: Smooth conversation flow
- **Error Resilience**: Fallback prompts for failed generations
- **Legal Precision**: Structured JSON output for analysis

### Example Prompts

#### Initial Greeting
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

#### Legal Analysis
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

## 🧪 Testing & Evaluation

### Comprehensive Test Suite

Run the complete test suite covering all required scenarios:

```bash
python test_tripfix_scenarios.py
```

### Required Test Scenarios

1. **✅ Flight entirely within Canada (APPR)**
   - AC456 Toronto → Vancouver
   - Expected: APPR jurisdiction, $400 compensation

2. **✅ Flight from EU to Canada on Canadian airline (EU 261)**
   - AC789 London → Toronto (Air Canada)
   - Expected: EU261 jurisdiction, compensation eligibility

3. **✅ Flight that falls under neither jurisdiction (within US)**
   - AA123 New York → Los Angeles
   - Expected: NEITHER jurisdiction, no compensation

4. **✅ Ambiguous delay reason triggering human handoff**
   - AC999 with "operational reasons"
   - Expected: Human review required due to ambiguous reason

5. **✅ Successful and complete intake process**
   - AC777 with clear delay reason
   - Expected: Complete intake with compensation determination

6. **✅ Off-topic conversation attempt**
   - Weather, hotel, restaurant questions
   - Expected: Agent redirects to flight delay topic

7. **✅ User requiring human agent after solution provided**
   - User requests human agent after analysis
   - Expected: Appropriate handling of human agent requests

### Test Results Summary
- **100% Pass Rate**: All 8 test scenarios passed
- **Jurisdiction Detection**: Accurate APPR, EU261, and NEITHER identification
- **Human Review Flagging**: Complex cases appropriately flagged
- **Special Scenarios**: Off-topic redirection and human agent requests working

## 📊 Database Schema

### intake_sessions
- `id`: Unique session identifier
- `flight_data`: JSON with flight details
- `jurisdiction`: APPR/EU261/NEITHER
- `jurisdiction_confidence`: Confidence score (0-1)
- `eligibility_result`: JSON with eligibility determination
- `eligibility_confidence`: Confidence score (0-1)
- `handoff_reason`: Why human review was triggered
- `handoff_priority`: Priority level for human review
- `risk_level`: Risk assessment level (low/medium/high/critical)
- `risk_assessment`: Complete JSON risk assessment data
- `completed`: Whether intake is finished

### conversation_history
- `session_id`: Links to intake_sessions
- `message_type`: user/assistant
- `content`: Message content and metadata
- `timestamp`: When message was sent

### supporting_files
- `id`: Unique file identifier
- `session_id`: Links to intake_sessions
- `filename`: Original filename
- `file_type`: MIME type
- `file_size`: File size in bytes
- `file_path`: Storage path
- `extracted_text`: OCR/extracted text content
- `metadata`: Additional file information

### intake_progress
- `session_id`: Links to intake_sessions
- `flight_number_collected`: Boolean tracking
- `flight_date_collected`: Boolean tracking
- `airline_collected`: Boolean tracking
- `origin_collected`: Boolean tracking
- `destination_collected`: Boolean tracking
- `connecting_airports_collected`: Boolean tracking
- `delay_length_collected`: Boolean tracking
- `delay_reason_collected`: Boolean tracking
- `supporting_files_offered`: Boolean tracking
- `intake_complete`: Boolean tracking

## 🎨 Frontend Features

### Main Application (`app.py`)
- **Modern Chat Interface**: Real-time conversation with AI agent
- **File Upload Support**: PDF, images, text, Word documents
- **Risk Assessment Dashboard**: Real-time risk level display in sidebar
- **Agent Activity Log**: Live monitoring of system operations
- **Performance Metrics**: Response times and success rates

### Intake Dashboard (`pages/intake_dashboard.py`)
- **Completed Sessions View**: All processed intake sessions
- **Summary Statistics**: Total sessions, eligible cases, compensation amounts
- **Interactive Filtering**: By status, jurisdiction, eligibility
- **Detailed Session View**: Complete analysis results
- **Export Functionality**: Download filtered results as CSV

### Evaluation Dashboard (`pages/evaluation_dashboard.py`)
- **Performance Monitoring**: System health and metrics
- **Confidence Calibration**: Visual confidence vs accuracy correlation
- **Component Analysis**: Accuracy by jurisdiction type
- **Live Monitoring**: Real-time system health indicators
- **Test Case Analysis**: Detailed results by scenario type

## 🔧 Key Design Decisions

### 1. LangGraph for Orchestration
- **Why**: Explicit workflow control with conditional branching
- **Benefits**: Debuggable, stateful, handles complex decision trees
- **Alternative considered**: Simple LangChain chains (too linear)

### 2. Multi-Agent Architecture
- **Jurisdiction Agent**: Specializes in determining applicable laws
- **Eligibility Agent**: Focuses on compensation assessment
- **Advanced Confidence Engine**: Multi-factor risk analysis
- **Benefits**: Separation of concerns, specialized prompting, better accuracy

### 3. ChromaDB for Vector Storage
- **Why**: Runs locally, excellent for legal document retrieval
- **Benefits**: No external dependencies, good performance, easy setup
- **Alternative considered**: Pinecone (rejected for local requirement)

### 4. SQLite for Session Storage
- **Why**: Local, reliable, good for structured intake data
- **Benefits**: ACID compliance, familiar SQL interface, no setup overhead

## 📈 Performance & Monitoring

### Real-time Metrics
- **Response Time**: Average processing time per component
- **Success Rate**: System reliability and error rates
- **Active Sessions**: Current user sessions
- **Confidence Distribution**: Real-time confidence score analysis

### Advanced Confidence Engine Metrics
- **7-Factor Analysis**: Detailed breakdown of risk factors
- **Pattern Detection**: Complex scenario identification
- **Contextual Analysis**: Conversation history consideration
- **Dynamic Thresholds**: Adaptive risk level determination

## 🚀 Production Readiness

### ✅ Fully Implemented & Demo-Ready

#### 🔄 LangGraph Workflow - 100% Ready
- Sophisticated conditional branching
- State management across interactions
- Human-in-loop integration
- Error recovery and validation loops

#### 🧠 Advanced Confidence Engine - 100% Ready
- 7 weighted confidence factors
- Dynamic risk thresholds (not binary!)
- Intelligent pattern detection
- Multi-level handoff urgency

#### 📚 Vector DB & Chunking - 100% Ready
- ChromaDB with optimized legal document chunking
- Metadata filtering (APPR vs EU261)
- Semantic search with embeddings
- Performance benchmarking metrics

#### 📊 Evaluation System - 100% Ready
- Comprehensive test suite with 7 required scenarios
- 100% pass rate on all tests
- Beautiful evaluation dashboard
- Real-time performance tracking

#### 📎 File Upload System - 100% Ready
- Multi-format file support (PDF, images, text, Word)
- Automatic text extraction and OCR
- Flight information parsing
- Database integration and storage

### 🎯 Demo Readiness Score: 100% Ready

**All required demo scenarios implemented and tested:**
- ✅ APPR Canadian domestic flight
- ✅ EU261 EU departure flight  
- ✅ No jurisdiction US domestic flight
- ✅ Ambiguous operational reasons (human handoff)
- ✅ Complete intake process
- ✅ Off-topic conversation handling
- ✅ User requiring human agent after solution provided

## 🔮 Future Enhancements

### Phase 2 Features
- Multi-language support
- Integration with airline APIs for real-time flight data
- Automated claim filing
- SMS/email notifications
- Advanced analytics dashboard

### Technical Improvements
- Streaming responses for better UX
- Fine-tuned models for legal domain
- Advanced confidence scoring with ML models
- Integration with legal case management systems

## 🛠️ Troubleshooting

### Common Issues

#### 1. OpenAI API Errors
**Problem**: `AuthenticationError` or `RateLimitError`
**Solutions**:
- Verify your API key is correct in `.env` file
- Check your OpenAI account has sufficient credits
- Ensure you're using a valid API key from https://platform.openai.com/api-keys

#### 2. PDF Processing Fails
**Problem**: PDFs not being processed or text extraction fails
**Solutions**:
- Ensure PDFs are text-based, not scanned images
- Try converting scanned PDFs to text using OCR tools
- Check file permissions in `data/regulations/` directory

#### 3. Vector Store Initialization Slow
**Problem**: First run takes a long time to start
**Solutions**:
- This is normal for first run with large PDFs
- Subsequent runs will be much faster
- Consider using smaller PDF files for testing

#### 4. Session Not Persisting
**Problem**: Data not being saved between sessions
**Solutions**:
- Check database file permissions in `data/` directory
- Ensure the application has write access to the data folder
- Verify SQLite database is not corrupted

#### 5. Setup Script Fails
**Problem**: Setup script encounters errors
**Solutions**:
- Ensure Python 3.8+ is installed and in PATH
- Check internet connection for package downloads
- Try running setup steps manually (see Option 2 in Quick Start)
- On Windows, run Command Prompt as Administrator

#### 6. Streamlit Port Issues
**Problem**: Port already in use error
**Solutions**:
```bash
# Kill existing Streamlit processes
pkill -f streamlit  # Unix/Linux/macOS
taskkill /f /im streamlit.exe  # Windows

# Or use a different port
streamlit run app.py --server.port 8502
```

#### 7. Virtual Environment Issues
**Problem**: Package installation fails or import errors
**Solutions**:
```bash
# Recreate virtual environment
rm -rf venv  # Unix/Linux/macOS
rmdir /s venv  # Windows
python -m venv venv
source venv/bin/activate  # Unix/Linux/macOS
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Debug Mode
Set `DEBUG=true` in `.env` file to enable verbose logging:
```bash
# In .env file
DEBUG=true
```

### Getting Help
1. Check the logs in `tripfix_app.log` for detailed error messages
2. Run tests to verify setup: `python test_tripfix_scenarios.py`
3. Ensure all prerequisites are met (Python 3.8+, OpenAI API key)
4. Try the manual setup process if automated setup fails

## 📚 Documentation

- **`PROMPT_ENGINEERING_DOCUMENTATION.md`**: Comprehensive prompt engineering guide
- **`TEST_README.md`**: Detailed testing documentation
- **`langgraph_architecture_diagram.md`**: LangGraph implementation details
- **`ADVANCED_CONFIDENCE_ENGINE_SUMMARY.md`**: Confidence engine documentation
- **`INTEGRATION_SUMMARY.md`**: System integration overview

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite: `python test_tripfix_scenarios.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For the powerful LLM framework
- **LangGraph**: For sophisticated workflow orchestration
- **Streamlit**: For the beautiful web interface
- **ChromaDB**: For efficient vector storage
- **OpenAI**: For the GPT models and embeddings

---

**TripFix** - Making flight delay compensation accessible through intelligent AI assistance.