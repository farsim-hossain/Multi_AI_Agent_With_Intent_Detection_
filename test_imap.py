import imaplib

# Replace these with your actual Gmail credentials
username = "farsimhossain@gmail.com"
app_password = "ixlm tgeb pvlo vooi"

try:
    # Connect to Gmail's IMAP server
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    print("Connecting to IMAP server...")
    
    # Attempt to log in
    response = mail.login(username, app_password)
    print(f"IMAP login successful! Response: {response}")
    mail.logout()
except Exception as e:
    print(f"IMAP login failed: {str(e)}")