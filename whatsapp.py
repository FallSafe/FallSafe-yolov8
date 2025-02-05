def send_whatsapp_alert(tonumber):
    try:
        from twilio.rest import Client
        from dotenv import load_dotenv
        import os
        load_dotenv()

        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        client = Client(account_sid, auth_token)

        to_number = 'whatsapp:'+tonumber
        print(to_number)
        from_number = os.getenv('SENDER_WHATSAPP_NUMBER')
        message_body = 'Fall Detected - Check Email for more information'

        message = client.messages.create(
            from_=from_number,
            body=message_body,
            to=to_number
        )
        print(f"WhatsApp alert sent (SID: {message.sid})")
        return True
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return False

# if __name__ == "__main__":
#     send_whatsapp_alert()