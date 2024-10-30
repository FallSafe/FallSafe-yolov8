import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
load_dotenv()

subject = 'Fall Detected'
body = 'Fall Detected!! Ceck your app for more details'

msg = MIMEMultipart()
msg['From'] = os.environ.get('SENDER_EMAIL')
msg['To'] = os.environ.get('RECIEVER_EMAIL')
msg['Subject'] = subject
msg.attach(MIMEText(body, 'plain'))

try:
    server = smtplib.SMTP(os.environ.get('SMTP_SERVER'), os.environ.get('SMTP_PORT'))
    server.starttls()
    server.login(os.environ.get('SENDER_EMAIL'), os.environ.get('SENDER_PASSWORD'))
    server.send_message(msg)
    print('Email sent successfully!')
except Exception as e:
    print(f'Error occurred: {e}')
finally:
    server.quit()
