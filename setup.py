from setuptools import setup, find_packages

setup(
    name="tripfix-intake",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.1",
        "langchain>=0.1.0",
        "langgraph>=0.0.20",
        "openai>=1.3.0",
        "chromadb>=0.4.18",
        "PyPDF2>=3.0.1",
        "python-dotenv>=1.0.0",
        "pandas>=2.1.4",
        "numpy>=1.24.3",
        "tiktoken>=0.5.1",
        "streamlit-chat>=0.1.1",
    ],
)