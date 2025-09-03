@echo off
REM TripFix AI Intake System Setup Script for Windows
REM This script automates the setup process for Windows systems

echo 🚀 TripFix AI Intake System Setup
echo ==================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python is installed

REM Create virtual environment
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo 📁 Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
if exist "requirements.txt" (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    echo ✅ Dependencies installed
) else (
    echo ❌ requirements.txt not found
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist ".env" (
    if exist "env.template" (
        echo 📄 Creating .env file from template...
        copy env.template .env
        echo ✅ .env file created from template
    ) else (
        echo 📄 Creating .env file...
        (
            echo # TripFix AI Intake System Environment Variables
            echo # Add your actual API keys here
            echo.
            echo # OpenAI API Key ^(required^)
            echo OPENAI_API_KEY=your_openai_api_key_here
            echo.
            echo # Optional: Debug mode
            echo DEBUG=false
        ) > .env
        echo ✅ .env file created
    )
    echo ⚠️  Please edit .env file and add your OpenAI API key
) else (
    echo 📄 .env file already exists
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "data" mkdir data
if not exist "data\uploads" mkdir data\uploads
if not exist "data\vectorstore" mkdir data\vectorstore
if not exist "data\regulations" mkdir data\regulations
if not exist "logs" mkdir logs
echo ✅ Directories created

REM Check for sample regulation files
echo 📚 Checking for sample regulation files...
if not exist "data\regulations\APPR Canada.pdf" (
    echo ⚠️  Sample regulation files not found
    echo 💡 Please add APPR Canada.pdf and EU.pdf to data\regulations\ for full functionality
) else (
    echo ✅ Sample regulation files found
)

REM Run initial tests
if exist "test_tripfix_scenarios.py" (
    echo 🧪 Running initial tests...
    python test_tripfix_scenarios.py
    if errorlevel 1 (
        echo ⚠️  Some tests failed - this may be due to missing API keys
        echo 💡 Complete the .env file setup and try again
    ) else (
        echo ✅ All tests passed!
    )
) else (
    echo ⚠️  Test file not found - skipping tests
)

echo.
echo 🎉 Setup Complete!
echo ==================
echo.
echo 📋 Next Steps:
echo 1. Edit .env file and add your OpenAI API key
echo 2. Activate the virtual environment: venv\Scripts\activate.bat
echo 3. Run the application: streamlit run app.py
echo.
echo 📚 Additional Commands:
echo    • Intake Dashboard: streamlit run pages\intake_dashboard.py
echo    • Evaluation Dashboard: streamlit run pages\evaluation_dashboard.py
echo    • Run Tests: python test_tripfix_scenarios.py
echo.
echo 🆘 Need Help? Check the README.md for detailed instructions
echo.
pause
