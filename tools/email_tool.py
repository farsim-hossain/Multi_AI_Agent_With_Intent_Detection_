import os
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from utils.env_loader import load_env
from dotenv import load_dotenv
from langchain_groq import ChatGroq

class EmailTool:
    def __init__(self, llm=None):
        load_dotenv()
        self.username = os.getenv('GMAIL_USERNAME')
        self.app_password = os.getenv('GMAIL_APP_PASSWORD')
        
        if not self.username or not self.app_password:
            raise ValueError("Gmail credentials not found in environment variables")
            
        self.llm = llm or ChatGroq(
            model="llama-3.2-90b-vision-preview", 
            api_key=os.getenv("GROQ_API_KEY")
        )


    def send_email(self, to, subject, body):
        try:
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(self.username, self.app_password)
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = to
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            server.sendmail(self.username, to, msg.as_string())
            server.quit()
            return f"Email sent to {to}."
        except Exception as e:
            return f"Failed to send email: {str(e)}"

    def read_emails(self, num_emails=5):
        try:
            imap_server = "imap.gmail.com"
            mail = imaplib.IMAP4_SSL(imap_server)
            mail.login(self.username, self.app_password)
            mail.select("INBOX")
            status, messages = mail.search(None, "ALL")
            if status != "OK":
                return "No emails found."
            
            email_ids = messages[0].split()[-num_emails:]
            emails = []
            for email_id in email_ids:
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                if status != "OK":
                    continue
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject = msg["subject"]
                        from_ = msg["from"]
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    body = part.get_payload(decode=True).decode()
                        else:
                            body = msg.get_payload(decode=True).decode()
                        emails.append({"from": from_, "subject": subject, "body": body})
            mail.logout()
            return emails
        except Exception as e:
            return f"Failed to read emails: {str(e)}"

    def generate_and_send_response(self, email_content, context):
        try:
            # Generate response using LLM
            prompt = f"Email content:\n{email_content['body']}\n\nContext provided:\n{context}\n\nGenerate a concise reply:"
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            generated_response = response.content.strip()

            # Extract sender's email address
            sender_email = email.utils.parseaddr(email_content["from"])[1]

            # Send the reply
            subject = f"Re: {email_content['subject']}"
            return self.send_email(sender_email, subject, generated_response)
        except Exception as e:
            return f"Failed to generate/send response: {str(e)}"