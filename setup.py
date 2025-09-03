#!/usr/bin/env python3
"""
TripFix AI Intake System Setup Script

This script automates the setup process for the TripFix AI Intake System.
It handles environment setup, dependency installation, and initial configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description, check=True):
    """Run a shell command with error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")

def create_virtual_environment():
    """Create and activate virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return
    
    print("📦 Creating virtual environment...")
    run_command(f"{sys.executable} -m venv venv", "Creating virtual environment")
    
    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_script = "venv/bin/activate"
        pip_path = "venv/bin/pip"
    
    print(f"✅ Virtual environment created at {venv_path}")
    print(f"💡 To activate: source {activate_script} (Unix) or {activate_script} (Windows)")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Determine pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_path = "venv/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_path} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if Path("requirements.txt").exists():
        run_command(f"{pip_path} install -r requirements.txt", "Installing requirements")
    else:
        print("❌ requirements.txt not found")
        sys.exit(1)

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_template = Path(".env.template")
    
    if env_file.exists():
        print("📄 .env file already exists")
        return
    
    if env_template.exists():
        print("📄 Creating .env file from template...")
        shutil.copy(env_template, env_file)
        print("✅ .env file created from template")
        print("⚠️  Please edit .env file and add your OpenAI API key")
    else:
        print("📄 Creating .env file...")
        with open(env_file, 'w') as f:
            f.write("# TripFix AI Intake System Environment Variables\n")
            f.write("# Copy this file and add your actual API keys\n\n")
            f.write("# OpenAI API Key (required)\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n\n")
            f.write("# Optional: Debug mode\n")
            f.write("DEBUG=false\n")
        print("✅ .env file created")
        print("⚠️  Please edit .env file and add your OpenAI API key")

def create_directories():
    """Create necessary directories."""
    print("📁 Creating necessary directories...")
    
    directories = [
        "data",
        "data/uploads",
        "data/vectorstore",
        "data/regulations",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def download_sample_data():
    """Download sample regulation files if they don't exist."""
    print("📚 Checking for sample regulation files...")
    
    regulations_dir = Path("data/regulations")
    sample_files = ["APPR Canada.pdf", "EU.pdf"]
    
    missing_files = []
    for file in sample_files:
        if not (regulations_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing regulation files: {', '.join(missing_files)}")
        print("💡 Please add these files to data/regulations/ for full functionality")
    else:
        print("✅ All sample regulation files found")

def run_initial_tests():
    """Run initial tests to verify setup."""
    print("🧪 Running initial tests...")
    
    # Determine python path based on OS
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_path = "venv/bin/python"
    
    if Path("test_tripfix_scenarios.py").exists():
        result = run_command(f"{python_path} test_tripfix_scenarios.py", "Running test suite", check=False)
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("⚠️  Some tests failed - this may be due to missing API keys")
            print("💡 Complete the .env file setup and try again")
    else:
        print("⚠️  Test file not found - skipping tests")

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 TripFix AI Intake System Setup Complete!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    print("3. Run the application:")
    print("   streamlit run app.py")
    print("\n📚 Additional Commands:")
    print("   • Intake Dashboard: streamlit run pages/intake_dashboard.py")
    print("   • Evaluation Dashboard: streamlit run pages/evaluation_dashboard.py")
    print("   • Run Tests: python test_tripfix_scenarios.py")
    print("\n📖 Documentation:")
    print("   • README.md - Complete setup and usage guide")
    print("   • PROMPT_ENGINEERING_DOCUMENTATION.md - Advanced prompt engineering")
    print("\n🆘 Need Help?")
    print("   • Check the README.md for detailed instructions")
    print("   • Review the troubleshooting section")
    print("   • Ensure your OpenAI API key is valid and has sufficient credits")
    print("\n" + "="*60)

def main():
    """Main setup function."""
    print("🚀 TripFix AI Intake System Setup")
    print("="*40)
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Create .env file
    create_env_file()
    
    # Create necessary directories
    create_directories()
    
    # Check for sample data
    download_sample_data()
    
    # Run initial tests
    run_initial_tests()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()