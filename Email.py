import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
load_dotenv()

def send_email_alert(label, confidence_score, receiver_email, frame_path=None):
    """
    Sends an email alert when a fall is detected, with an optional frame attachment.
    :param label: The label associated with the detected event (e.g., 'Fall Detected').
    :param confidence_score: The confidence score of the detection.
    :param receiver_email: The recipient email address to send the alert.
    :param frame_path: Path to the captured frame image to attach.
    """
    try:
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", 465))

        if not sender_email or not sender_password:
            raise ValueError("Missing sender email or password in environment variables.")

        if not receiver_email:
            raise ValueError("Receiver email is not provided.")

        # Email content
        subject = f"Alert: {label}"
        body = f"A fall was detected with a confidence score of {confidence_score:.2f}. Please check the attached frame for details."

        # Create email message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        # Attach the frame if provided
        if frame_path and os.path.exists(frame_path):
            with open(frame_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={os.path.basename(frame_path)}",
                )
                message.attach(part)

        # Send email via SMTP server
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print(f"Email sent successfully to {receiver_email}")
            return f"Alert sent to {receiver_email}."

    except Exception as e:
        print(f"Error sending email: {e}")
        return f"Error: {e}"
