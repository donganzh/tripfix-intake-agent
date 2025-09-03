#!/bin/bash

# TripFix AI Intake System Setup Script
# This script automates the setup process for Unix/Linux/macOS systems

set -e  # Exit on any error

echo "🚀 TripFix AI Intake System Setup"
echo "=================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION is compatible"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "📁 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f "env.template" ]; then
        echo "📄 Creating .env file from template..."
        cp env.template .env
        echo "✅ .env file created from template"
    else
        echo "📄 Creating .env file..."
        cat > .env << EOF
# TripFix AI Intake System Environment Variables
# Add your actual API keys here

# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Debug mode
DEBUG=false
EOF
        echo "✅ .env file created"
    fi
    echo "⚠️  Please edit .env file and add your OpenAI API key"
else
    echo "📄 .env file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/uploads
mkdir -p data/vectorstore
mkdir -p data/regulations
mkdir -p logs
echo "✅ Directories created"

# Check for sample regulation files
echo "📚 Checking for sample regulation files..."
if [ ! -f "data/regulations/APPR Canada.pdf" ] || [ ! -f "data/regulations/EU.pdf" ]; then
    echo "⚠️  Sample regulation files not found"
    echo "💡 Please add APPR Canada.pdf and EU.pdf to data/regulations/ for full functionality"
else
    echo "✅ Sample regulation files found"
fi

# Run initial tests
if [ -f "test_tripfix_scenarios.py" ]; then
    echo "🧪 Running initial tests..."
    if python test_tripfix_scenarios.py; then
        echo "✅ All tests passed!"
    else
        echo "⚠️  Some tests failed - this may be due to missing API keys"
        echo "💡 Complete the .env file setup and try again"
    fi
else
    echo "⚠️  Test file not found - skipping tests"
fi

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the application: streamlit run app.py"
echo ""
echo "📚 Additional Commands:"
echo "   • Intake Dashboard: streamlit run pages/intake_dashboard.py"
echo "   • Evaluation Dashboard: streamlit run pages/evaluation_dashboard.py"
echo "   • Run Tests: python test_tripfix_scenarios.py"
echo ""
echo "🆘 Need Help? Check the README.md for detailed instructions"
