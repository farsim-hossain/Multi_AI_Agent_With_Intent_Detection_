import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    required_vars = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GMAIL_CREDENTIALS_PATH": os.getenv("GMAIL_CREDENTIALS_PATH"),
        "GMAIL_USERNAME": os.getenv("GMAIL_USERNAME"),
        "GMAIL_APP_PASSWORD": os.getenv("GMAIL_APP_PASSWORD")
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
    return tuple(required_vars.values())
