import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")  # Default to false if not set
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "default_project")

# Ensure required variables are loaded
REQUIRED_VARS = ["LANGCHAIN_API_KEY"]
for var in REQUIRED_VARS:
    if not globals()[var]:
        raise ValueError(f"Missing required environment variable: {var}")
