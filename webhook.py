# Updated webhook.py - CNIC-based authentication with account selection + EXIT functionality
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
import os
import requests
import re
from typing import Dict, Any
from state import (
    authenticated_users, processed_messages, periodic_cleanup,
    VERIFICATION_STAGES, get_user_verification_stage, set_user_verification_stage,
    is_fully_authenticated, get_user_account_info, clear_user_state
)
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

VERIFY_TOKEN = "helloworld3"
PAGE_ACCESS_TOKEN = "EAAOqZBb1DZCWYBPGiZCdRaVk6KrTAiQYclW4ZCZC9e8FiC4EqdOU0zN2gLDaVC1UtXeDXYT7VtnKPyr5NV3TZAgChtsMiDhzgZBsqk6eHZA8IKUQjqlORPXIatiTbs9OekNOeFxL16xOpEM2gJKMgJLR7yo70dPCHWBTyILXZAiBLEzQt9KfZBdOYCIEGyOVDdzMDM9aey"

BACKEND_URL = "http://localhost:8000"

class ProfessionalResponseFormatter:
    """Handles professional, warm response formatting for Sage banking assistant."""
    
    def __init__(self):
        self.assistant_name = "Sage"
        
    def get_time_of_day_greeting(self):
        """Get appropriate greeting based on time of day."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        elif 17 <= hour < 21:
            return "Good evening"
        else:
            return "Hello"

# Initialize formatter
response_formatter = ProfessionalResponseFormatter()

def get_welcome_message() -> str:
    """Welcome message for new users."""
    time_greeting = response_formatter.get_time_of_day_greeting()
    
    return f"""{time_greeting}! Welcome to Sage, your personal banking assistant. 🏦✨

I'm here to help you manage your banking needs securely and efficiently.

To get started, I'll need to verify your identity using your CNIC (National Identity Card number).

**Please enter your CNIC in this format:**
42501-1234567-8

Once verified, you'll be able to:
💰 Check account balances
📊 Analyze spending patterns  
📝 View transaction history
💸 Transfer money securely
🤖 Have natural conversations about your finances

Please share your CNIC to begin!"""

def get_session_terminated_message() -> str:
    """Session termination message."""
    return """🔐 **Session Terminated**

Your banking session has been safely ended for security.

To start a new session, please provide your CNIC number in the format: 12345-1234567-1

---
*Type 'exit' anytime to end your session*"""

def get_account_selection_message(name: str, accounts: list) -> str:
    """Account selection message after CNIC verification."""
    
    accounts_text = ""
    for i, account in enumerate(accounts, 1):
        # Format account number for display (show last 4 digits)
        formatted_account = f"***-***-{account[-4:]}"
        accounts_text += f"**{i}.** Account {formatted_account}\n"
    
    return f"""Great news, {name}! ✅ Your CNIC has been verified successfully.

I found {len(accounts)} accounts associated with your CNIC. Please select which account you'd like to access:

{accounts_text}

**To select an account, simply reply with the number (1 or 2).**

Once you select an account, I'll remember our entire conversation and you can ask me questions like:
• "What's my current balance?"
• "Show me last month's spending"
• "Transfer 5000 PKR to Ahmed"

**Security:** Type 'exit' anytime to safely end your session.

Which account would you like to access?"""

def get_account_confirmed_message(name: str, account: str) -> str:
    """Account selection confirmation message."""
    time_greeting = response_formatter.get_time_of_day_greeting()
    formatted_account = f"***-***-{account[-4:]}"
    
    return f"""Perfect! ✅ Account {formatted_account} selected successfully.

{time_greeting}, {name}! I'm now ready to assist you with your banking needs. 

I can help you with:

💰 **Account Information**
• Check your current balance
• Review recent account activity

📊 **Financial Analysis** 
• Analyze your spending patterns
• Break down expenses by category
• Track spending trends over time

📝 **Transaction Services**
• View your transaction history
• Search and filter transactions
• Transfer money securely

I'm designed to understand context and remember our conversation, so you can have natural conversations with me. For example, you can ask "How much did I spend last month?" and then follow up with "from this, how much on food?"

**Security:** Type 'exit' anytime to safely end your banking session.

What would you like to know about your account today?"""

@app.get("/webhook")
async def webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge, status_code=200)
    else:
        raise HTTPException(status_code=403, detail="Invalid verification token.")

@app.post("/webhook")
async def receive_message(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)
    
    if "entry" not in data:
        return JSONResponse(content={"status": "ok"})

    for entry in data.get("entry", []):
        for messaging_event in entry.get("messaging", []):
            message_id = messaging_event.get("message", {}).get("mid")
            sender_id = messaging_event["sender"]["id"]

            if message_id and message_id in processed_messages:
                continue

            if "message" in messaging_event:
                if message_id:
                    processed_messages.add(message_id)
                
                user_message = messaging_event["message"].get("text", "")
                
                if user_message.strip():
                    response_text = await process_user_message(sender_id, user_message)
                    send_message(sender_id, response_text)

    if len(processed_messages) % 100 == 0:
        periodic_cleanup()

    return JSONResponse(content={"status": "ok"})

user_last_message_time = {}

async def process_user_message(sender_id: str, user_message: str) -> str:
    """Process user message with CNIC-based authentication flow + EXIT command."""
    
    current_time = time.time()
    
    # Rate limiting
    if sender_id in user_last_message_time:
        if current_time - user_last_message_time[sender_id] < 2:
            return "I appreciate your enthusiasm! Please give me just a moment to process your previous message before sending another. 😊"
    
    user_last_message_time[sender_id] = current_time

    # 🚪 CHECK FOR EXIT COMMAND FIRST (before any other processing)
    if user_message.strip().lower() == "exit":
        logger.info({
            "action": "exit_command_detected",
            "sender_id": sender_id
        })
        
        # Clear user session completely
        clear_user_state(sender_id)
        
        logger.info({
            "action": "session_terminated",
            "sender_id": sender_id
        })
        
        return get_session_terminated_message()

    # Get current verification stage
    verification_stage = get_user_verification_stage(sender_id)
    
    logger.info({
        "action": "processing_user_message",
        "sender_id": sender_id,
        "verification_stage": verification_stage,
        "user_message": user_message
    })

    # Handle different verification stages
    if verification_stage == VERIFICATION_STAGES["NOT_VERIFIED"]:
        return await handle_cnic_verification(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["CNIC_VERIFIED"]:
        return await handle_account_selection(sender_id, user_message)
    
    elif verification_stage == VERIFICATION_STAGES["ACCOUNT_SELECTED"]:
        return await handle_banking_queries(sender_id, user_message)
    
    else:
        # Fallback to welcome message
        return get_welcome_message()

async def handle_cnic_verification(sender_id: str, user_message: str) -> str:
    """Handle CNIC verification step."""
    
    # Check if message looks like a CNIC
    cnic_pattern = r'^\d{5}-\d{7}-\d$'  # Fixed: Added missing closing quote
    user_message_clean = user_message.strip()
    
    if not re.match(cnic_pattern, user_message_clean):
        return """Please enter a valid CNIC number in the correct format:

**Format:** 42501-1234567-8

Your CNIC should be:
• 5 digits, then dash (-)
• 7 digits, then dash (-)  
• 1 digit

Please try again with your complete CNIC number."""
    
    try:
        # Verify CNIC with backend
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/verify_cnic",
                json={"cnic": user_message_clean}
            )
            result = response.json()
        
        if result["status"] == "success":
            user_data = result["user"]
            
            # Store CNIC verification data
            set_user_verification_stage(
                sender_id, 
                VERIFICATION_STAGES["CNIC_VERIFIED"],
                cnic=user_data["cnic"],
                name=user_data["name"],
                accounts=user_data["accounts"]
            )
            
            logger.info({
                "action": "cnic_verified_successfully",
                "sender_id": sender_id,
                "cnic": user_data["cnic"],
                "name": user_data["name"],
                "accounts_count": len(user_data["accounts"])
            })
            
            # Return account selection message
            return get_account_selection_message(user_data["name"], user_data["accounts"])
        
        else:
            logger.warning({
                "action": "cnic_verification_failed",
                "sender_id": sender_id,
                "cnic": user_message_clean,
                "reason": result.get("reason", "Unknown")
            })
            
            return """❌ CNIC verification failed.

The CNIC number you entered was not found in our system. Please check and try again.

**Make sure:**
• You've entered the correct CNIC number
• The format is: 12345-1234567-8
• All digits are correct

Please enter your CNIC again, or contact support if you continue to have issues."""
    
    except Exception as e:
        logger.error({
            "action": "cnic_verification_error",
            "sender_id": sender_id,
            "error": str(e)
        })
        
        return """I encountered a technical issue while verifying your CNIC. Please try again in a moment.

If the problem persists, please contact our support team."""

async def handle_account_selection(sender_id: str, user_message: str) -> str:
    """Handle account selection step."""
    
    user_data = authenticated_users[sender_id]
    accounts = user_data.get("accounts", [])
    
    # Check if user selected a valid account number
    selection = user_message.strip()
    
    if selection in ["1", "2"]:
        try:
            account_index = int(selection) - 1
            if 0 <= account_index < len(accounts):
                selected_account = accounts[account_index]
                
                # Verify account selection with backend
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{BACKEND_URL}/select_account",
                        json={
                            "cnic": user_data["cnic"],
                            "account_number": selected_account
                        }
                    )
                    result = response.json()
                
                if result["status"] == "success":
                    # Update to final verification stage
                    set_user_verification_stage(
                        sender_id,
                        VERIFICATION_STAGES["ACCOUNT_SELECTED"],
                        cnic=user_data["cnic"],
                        name=user_data["name"],
                        selected_account=selected_account
                    )
                    
                    logger.info({
                        "action": "account_selected_successfully",
                        "sender_id": sender_id,
                        "cnic": user_data["cnic"],
                        "selected_account": selected_account
                    })
                    
                    return get_account_confirmed_message(user_data["name"], selected_account)
                
                else:
                    return "Account selection failed. Please try again or contact support."
            
            else:
                return f"""Please select a valid option.

You have {len(accounts)} accounts available. Reply with:
• **1** for the first account
• **2** for the second account

**Security:** Type 'exit' to end your session."""
        
        except Exception as e:
            logger.error({
                "action": "account_selection_error",
                "sender_id": sender_id,
                "error": str(e)
            })
            return "Account selection failed. Please try again."
    
    else:
        return f"""Please select an account by entering the number.

Reply with:
• **1** for the first account  
• **2** for the second account

Your available accounts:
{chr(10).join([f"{i+1}. Account ***-***-{acc[-4:]}" for i, acc in enumerate(accounts)])}

**Security:** Type 'exit' to end your session."""

async def handle_banking_queries(sender_id: str, user_message: str) -> str:
    """Handle banking queries for fully authenticated users."""
    
    user_info = get_user_account_info(sender_id)
    account_number = user_info["account_number"]
    first_name = user_info["name"].split()[0]  # Get first name
    
    try:
        logger.info({
            "action": "processing_banking_query",
            "sender_id": sender_id,
            "account_number": account_number,
            "user_message": user_message
        })
        
        # Make API call to backend process_query endpoint
        response = await call_process_query_api(
            user_message=user_message,
            account_number=account_number,
            first_name=first_name
        )
        
        logger.info({
            "action": "banking_query_processed_successfully",
            "sender_id": sender_id,
            "response_length": len(response)
        })
        
        return response
        
    except Exception as e:
        logger.error({
            "action": "banking_query_error",
            "sender_id": sender_id,
            "error": str(e),
            "user_message": user_message
        })
        return f"I apologize, {first_name}, but I encountered a technical issue while processing your request. Please try again, and I'll be happy to help you!"

async def call_process_query_api(user_message: str, account_number: str, first_name: str) -> str:
    """Make API call to backend process_query endpoint."""
    try:
        payload = {
            "user_message": user_message,
            "account_number": account_number,
            "first_name": first_name
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/process_query",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result["status"] == "success":
                return result["response"]
            else:
                logger.error({
                    "action": "process_query_api_error",
                    "error": result.get("error", "Unknown error"),
                    "account_number": account_number
                })
                return result.get("response", "Sorry, I couldn't process your request. Please try again.")
                
    except httpx.TimeoutException:
        logger.error({
            "action": "process_query_api_timeout",
            "account_number": account_number,
            "user_message": user_message
        })
        return "Request timed out. Please try again with a simpler query."
        
    except httpx.HTTPStatusError as e:
        logger.error({
            "action": "process_query_api_http_error",
            "status_code": e.response.status_code,
            "account_number": account_number,
            "error": str(e)
        })
        return "Backend service error. Please try again later."
        
    except Exception as e:
        logger.error({
            "action": "process_query_api_unexpected_error",
            "error": str(e),
            "account_number": account_number
        })
        return "Unexpected error occurred. Please try again."

def send_message(recipient_id, message_text):
    """Send response to Facebook Messenger."""
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info({
            "action": "message_sent_successfully",
            "recipient_id": recipient_id,
            "response_status": response.status_code
        })
    except requests.exceptions.RequestException as e:
        logger.error({
            "action": "send_message_error",
            "recipient_id": recipient_id,
            "error": str(e)
        })

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            backend_healthy = response.status_code == 200
    except:
        backend_healthy = False
    
    return {
        "status": "healthy",
        "backend_connection": "healthy" if backend_healthy else "unhealthy",
        "timestamp": time.time(),
        "service": "banking_webhook_cnic",
        "authentication_flow": "cnic_verification_account_selection",
        "memory_system": "langchain_conversation_buffer",
        "exit_functionality": "enabled"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)