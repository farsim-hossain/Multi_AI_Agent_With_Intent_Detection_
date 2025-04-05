import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Define the scopes for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_credentials(credentials_path):
    creds = None
    
    # Check if token.pickle file exists (stores user credentials)
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials are available, perform the OAuth2 flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for future use
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

if __name__ == "__main__":
    # Path to your web application credentials file
    credentials_path = "utils/credentials.json"
    
    # Get user credentials
    creds = get_credentials(credentials_path)
    print("OAuth2 authentication completed. User credentials saved to token.pickle.")