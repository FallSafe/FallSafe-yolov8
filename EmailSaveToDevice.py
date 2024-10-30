import os
import hashlib
from imbox import Imbox
from datetime import datetime
import traceback
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Load and validate environment variables
host = os.getenv('IMAP_SERVER')
receiver_email = os.getenv("RECEIVER_EMAIL")
receiver_password = os.getenv("RECEIVER_PASSWORD")
download_folder = r"D:"
sender_email = os.getenv('SENDER_EMAIL')

# Ensure base download folder exists
if not os.path.isdir(download_folder):
    os.makedirs(download_folder, exist_ok=True)

# Function to check if a file exists
def is_duplicate(file_path, content):
    if os.path.exists(file_path):
        with open(file_path, "rb") as existing_file:
            existing_content = existing_file.read()
            return hashlib.md5(existing_content).hexdigest() == hashlib.md5(content).hexdigest()
    return False

try:
    # Connect to the mail server
    print("Connecting to the mail server...")
    with Imbox(host, username=receiver_email, password=receiver_password, ssl=True) as mail:
        print("Connected successfully.")

        duplicate_threshold = 1
        duplicate_folder_count = 0
        message_count = 0

        # Fetching messages one by one from the specified sender
        for (uid, message) in mail.messages(sent_from=sender_email):
            # Stop fetching if the consecutive duplicate folders threshold is hit
            if duplicate_folder_count >= duplicate_threshold:
                print("Exiting due to ",duplicate_threshold, "consecutive duplicate folders.")
                break
            
            message_count += 1
            print(f"Processing message UID: {uid} (Message {message_count})")
            mail.mark_seen(uid)  # Mark the message as read
            email_date = message.date
            if isinstance(email_date, str):
                email_date = re.sub(r"\s+\(.*?\)", "", email_date)
                try:
                    email_date = datetime.strptime(email_date, '%a, %d %b %Y %H:%M:%S %z')
                except ValueError:
                    print(f"Error parsing date: {email_date}")
                    continue

            # Format the folder name with date and time (YYYY-MM-DD_HH-MM-SS)
            formatted_date_time = email_date.strftime('%Y-%m-%d_%H-%M-%S')
            email_folder = os.path.join(download_folder, formatted_date_time)
            
            if os.path.exists(email_folder):
                print(f"Duplicate folder found: {email_folder}")
                duplicate_folder_count += 1
                continue
            else:
                print(f"Creating folder: {email_folder}")
                duplicate_folder_count = 0  # Reset counter if non-duplicate found
                os.makedirs(email_folder, exist_ok=True)
            
            # Download each attachment if it's not a duplicate
            for attachment in message.attachments:
                try:
                    filename = attachment.get('filename')
                    file_content = attachment.get('content').read()
                    download_path = os.path.join(email_folder, filename)
                    
                    # Check if file is a duplicate based on hash comparison
                    if not is_duplicate(download_path, file_content):
                        print(f"Downloading attachment: {filename} to {download_path}")
                        with open(download_path, "wb") as file:
                            file.write(file_content)
                        print(f"Downloaded: {download_path}")
                    else:
                        print(f"Skipped duplicate file: {download_path}")
                except Exception as e:
                    print(f"Error saving attachment {filename}: {e}")
except Exception as e:
    print("Failed to connect to mail server:", e)
