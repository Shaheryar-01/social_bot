# webhook.py - Pure API-Based Communication with Language Support
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
import os
import requests
import re
import asyncio
import time
from typing import Dict, Any
from state import (
    authenticated_users, processed_messages, periodic_cleanup,
    VERIFICATION_STAGES, get_user_verification_stage, set_user_verification_stage,
    is_fully_authenticated, get_user_account_info, clear_user_state
)
from language_utils import detect_user_language, get_localized_response
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

VERIFY_TOKEN = "helloworld3"
PAGE_ACCESS_TOKEN = "EAAOqZBb1DZCWYBPGiZCdRaVk6KrTAiQYclW4ZCZC9e8FiC4EqdOU0zN2gLDaVC1UtXeDXYT7VtnKPyr5NV3TZAgChtsMiDhzgZBsqk6eHZA8IKUQjqlORPXIatiTbs9OekNOeFxL16xOpEM2gJKMgJLR7yo70dPCHWBTyILXZAiBLEzQt9KfZBdOYCIEGyOVDdzMDM9aey"

BACKEND_URL = "http://localhost:8000"

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
                    # Enhanced thinking indicator for slow responses
                    start_time = time.time()
                    
                    # Show typing indicator immediately for complex queries
                    if await is_likely_complex_query(user_message):
                        send_typing_indicator(sender_id)
                    
                    # Process message with pure API-based approach (now with language support)
                    response_text = await process_user_message_api_based(sender_id, user_message)
                    
                    # Ensure response isn't too long for Facebook
                    response_text = _ensure_message_length(response_text)
                    
                    # Check if we need thinking indicator
                    processing_time = time.time() - start_time
                    if processing_time > 2.0:
                        # Send thinking message first, then actual response (with language support)
                        user_language = detect_user_language(user_message)
                        thinking_msg = get_localized_response("processing", user_language)
                        send_message(sender_id, thinking_msg)
                        await asyncio.sleep(0.5)  # Brief pause
                    
                    send_message(sender_id, response_text)

    # Periodic cleanup
    if len(processed_messages) % 100 == 0:
        periodic_cleanup()

    return JSONResponse(content={"status": "ok"})

async def is_likely_complex_query(message: str) -> bool:
    """Quick check to determine if query might be complex."""
    complex_indicators = [
        "compare", "analysis", "spending", "most expensive", "highest", "lowest",
        "average", "total", "between", "from last", "in mein se", "sab se", 
        "pattern", "trend", "category", "group by", "sabse zyada", "sabse kam"
    ]
    message_lower = message.lower()
    return any(indicator in message_lower for indicator in complex_indicators)

user_last_message_time = {}

def is_greeting_message(message: str) -> bool:
    """Check if the message is a greeting with language support."""
    greeting_words = [
        # English greetings
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "good day", "howdy", "what's up", "whats up", "sup",
        # Other language greetings
        "hola", "bonjour", "namaste", "salaam", "salam", "assalam", "start",
        # Urdu/Hindi greetings
        "aslam alaikum", "assalam alaikum", "adaab", "namaskar"
    ]
    
    message_lower = message.lower().strip()
    
    # Check if message is exactly a greeting or starts with greeting
    for greeting in greeting_words:
        if message_lower == greeting or message_lower.startswith(greeting + " "):
            return True
    
    # Check for common greeting patterns
    greeting_patterns = [
        r'^hi+$',  # hi, hii, hiii
        r'^hey+$',  # hey, heyy
        r'^hello+$',  # hello, helloo
        r'^good (morning|afternoon|evening|day)',
        r'^how are you',
        r'^what\'?s up'
    ]
    
    for pattern in greeting_patterns:
        if re.match(pattern, message_lower):
            return True
    
    return False

def _ensure_message_length(message_text):
    """Ensure message is appropriate length for Facebook Messenger."""
    MAX_SINGLE_MESSAGE = 1900  # Leave buffer for Facebook limits
    
    if len(message_text) <= MAX_SINGLE_MESSAGE:
        return message_text
    
    # If message is too long, it will be split by send_message function
    # But log this for monitoring
    logger.warning({
        "action": "long_message_detected",
        "message_length": len(message_text),
        "will_be_split": True
    })
    
    return message_text

async def process_user_message_api_based(sender_id: str, user_message: str) -> str:
    """Pure API-based message processing with language support - NO direct AI agent imports."""
    
    current_time = time.time()
    user_language = detect_user_language(user_message)
    
    # Rate limiting with language support
    if sender_id in user_last_message_time:
        if current_time - user_last_message_time[sender_id] < 1.5:
            return get_localized_response("rate_limit", user_language)
    
    user_last_message_time[sender_id] = current_time

    # 🚪 CHECK FOR EXIT COMMAND FIRST (with language support)
    exit_commands = ["exit", "bye", "goodbye", "quit", "end", "khatam", "alvida", "bye bye"]
    if user_message.strip().lower() in exit_commands:
        logger.info({
            "action": "exit_command_detected",
            "sender_id": sender_id,
            "user_language": user_language,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get user info for personalized goodbye
        user_info = get_user_account_info(sender_id)
        first_name = user_info.get("name", "").split()[0] if user_info.get("name") else ""
        account_number = user_info.get("account_number", "")
        
        # Clear user session completely
        clear_user_state(sender_id)
        
        # API call for session end
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_URL}/handle_session_end",
                    json={
                        "account_number": account_number,
                        "first_name": first_name,
                        "user_language": user_language
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", get_localized_response("session_ended", user_language))
                else:
                    return get_localized_response("session_ended", user_language)
        except Exception as e:
            logger.error(f"Session end API call failed: {e}")
            return get_localized_response("session_ended", user_language)

    # Get current verification stage
    verification_stage = get_user_verification_stage(sender_id)
    
    logger.info({
        "action": "processing_user_message_api_based",
        "sender_id": sender_id,
        "verification_stage": verification_stage,
        "user_language": user_language,
        "user_message": user_message[:100] + "..." if len(user_message) > 100 else user_message,
        "message_length": len(user_message)
    })

    # Handle different verification stages with API calls (now with language support)
    if verification_stage == VERIFICATION_STAGES["NOT_VERIFIED"]:
        return await handle_cnic_verification_api_based(sender_id, user_message, user_language)
    
    elif verification_stage == VERIFICATION_STAGES["CNIC_VERIFIED"]:
        return await handle_account_selection_api_based(sender_id, user_message, user_language)
    
    elif verification_stage == VERIFICATION_STAGES["ACCOUNT_SELECTED"]:
        return await handle_authenticated_queries_api_based(sender_id, user_message, user_language)
    
    else:
        # API call for initial greeting
        return await api_call_initial_greeting(user_language)

async def handle_cnic_verification_api_based(sender_id: str, user_message: str, user_language: str) -> str:
    """Handle CNIC verification with pure API calls and language support."""
    
    user_message_clean = user_message.strip()
    
    # 🔧 Check if this is a greeting first
    if is_greeting_message(user_message_clean):
        logger.info({
            "action": "initial_greeting_detected",
            "sender_id": sender_id,
            "user_language": user_language,
            "message": user_message_clean
        })
        
        # API call for initial choice detection
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BACKEND_URL}/detect_initial_choice",
                    json={
                        "user_message": user_message_clean,
                        "first_name": "there",
                        "user_language": user_language
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    choice_detected = result.get("choice_detected")
                    
                    if choice_detected == "1":
                        # RAG mode - API call for RAG response
                        rag_response = await client.post(
                            f"{BACKEND_URL}/rag_query",
                            json={
                                "user_message": "tell me about the bank",
                                "first_name": "there",
                                "user_language": user_language
                            }
                        )
                        if rag_response.status_code == 200:
                            return rag_response.json().get("response", get_localized_response("bank_info_help", user_language))
                        
                    elif choice_detected == "2":
                        return get_localized_response("account_access_start", user_language)
                    
                    else:
                        # Show initial choices
                        return await api_call_initial_greeting(user_language)
                        
        except Exception as e:
            logger.error(f"Initial choice API call failed: {e}")
            return await api_call_initial_greeting(user_language)
    
    # 🔧 Check if this is a bank information query (not authenticated yet)
    bank_info_keywords = [
        # English keywords
        "tell me about best bank", "tell me about the bank", "about best bank", 
        "bank information", "bank info", "best bank info", "bank details",
        "bank services", "what services", "bank hours", "about the bank", 
        "bank policies", "loan information", "credit card info",
        "what does best bank offer", "best bank features", "bank products",
        "how does best bank work", "best bank overview", "bank description",
        "give me best bank info", "i want best bank info", "best bank details",
        "what is best bank", "explain best bank", "describe best bank",
        
        # Roman Urdu keywords
        "bank ke baare mein", "bank ki information", "bank ki services", 
        "best bank kya hai", "bank ke hours", "bank ki policies",
        "bank ki madad", "services kya hain", "bank kaise kaam karta"
    ]
    
    user_message_lower = user_message_clean.lower()
    is_bank_info_query = any(keyword in user_message_lower for keyword in bank_info_keywords)
    
    if is_bank_info_query:
        logger.info({
            "action": "bank_info_query_before_auth",
            "sender_id": sender_id,
            "user_language": user_language,
            "message": user_message_clean
        })
        
        # Handle bank info query via RAG API
        try:
            async with httpx.AsyncClient() as client:
                rag_response = await client.post(
                    f"{BACKEND_URL}/rag_query",
                    json={
                        "user_message": user_message_clean,
                        "first_name": "there",
                        "user_language": user_language
                    }
                )
                if rag_response.status_code == 200:
                    rag_result = rag_response.json()
                    return rag_result.get("response", get_localized_response("bank_info_help", user_language))
                else:
                    return get_localized_response("bank_info_help", user_language)
        except Exception as e:
            logger.error(f"RAG API call failed: {e}")
            return get_localized_response("bank_info_help", user_language)
    
    # 🔧 Extract CNIC from text - API call
    extracted_cnic = await api_call_extract_cnic(user_message_clean)
    
    if extracted_cnic:
        cnic_to_verify = extracted_cnic
        logger.info({
            "action": "cnic_extracted_via_api",
            "sender_id": sender_id,
            "user_language": user_language,
            "original_text": user_message_clean,
            "extracted_cnic": extracted_cnic
        })
    else:
        # Check if it's a valid CNIC format directly
        cnic_pattern = r'^\d{5}-\d{7}-\d$'
        if re.match(cnic_pattern, user_message_clean):
            cnic_to_verify = user_message_clean
        else:
            # API call for invalid CNIC format handling
            return await api_call_handle_invalid_cnic(user_message_clean, user_language)
    
    try:
        # Verify CNIC with backend
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/verify_cnic",
                json={
                    "cnic": cnic_to_verify,
                    "user_language": user_language
                }
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
                "user_language": user_language,
                "cnic": user_data["cnic"],
                "name": user_data["name"],
                "accounts_count": len(user_data["accounts"])
            })
            
            # API call for verification success response
            return await api_call_cnic_verification_success(user_data, user_language)
        
        else:
            logger.warning({
                "action": "cnic_verification_failed",
                "sender_id": sender_id,
                "user_language": user_language,
                "cnic": cnic_to_verify,
                "reason": result.get("reason", "Unknown")
            })
            
            # API call for verification failure response
            return await api_call_cnic_verification_failure(cnic_to_verify, user_language)
    
    except Exception as e:
        logger.error({
            "action": "cnic_verification_error",
            "sender_id": sender_id,
            "user_language": user_language,
            "error": str(e)
        })
        
        # API call for error handling
        return await api_call_error_response(str(e), user_message_clean, "", user_language)

async def handle_account_selection_api_based(sender_id: str, user_message: str, user_language: str) -> str:
    """Handle account selection with pure API calls and language support."""
    
    user_data = authenticated_users[sender_id]
    accounts = user_data.get("accounts", [])
    first_name = user_data.get("name", "").split()[0]
    
    try:
        # API call for enhanced account selection
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/enhanced_account_selection",
                json={
                    "user_input": user_message.strip(),
                    "available_accounts": accounts,
                    "first_name": first_name,
                    "user_language": user_language
                }
            )
            
            if response.status_code == 200:
                selection_result = response.json()
                
                logger.info({
                    "action": "enhanced_account_selection_api",
                    "sender_id": sender_id,
                    "user_language": user_language,
                    "selection_method": selection_result.get("selection_method"),
                    "matched_account": selection_result.get("matched_account"),
                    "user_input": user_message.strip()
                })
                
                if selection_result["matched_account"]:
                    selected_account = selection_result["matched_account"]
                    
                    # Verify account selection with backend
                    verify_response = await client.post(
                        f"{BACKEND_URL}/select_account",
                        json={
                            "cnic": user_data["cnic"],
                            "account_number": selected_account,
                            "user_language": user_language
                        }
                    )
                    verify_result = verify_response.json()
                    
                    if verify_result["status"] == "success":
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
                            "user_language": user_language,
                            "cnic": user_data["cnic"],
                            "selected_account": selected_account,
                            "selection_method": selection_result["selection_method"]
                        })
                        
                        # API call for account confirmation response
                        confirmation_response = await client.post(
                            f"{BACKEND_URL}/handle_account_confirmation",
                            json={
                                "selected_account": selected_account,
                                "user_name": user_data["name"],
                                "user_language": user_language
                            }
                        )
                        
                        if confirmation_response.status_code == 200:
                            return confirmation_response.json().get("response", 
                                f"Perfect! Account ***-***-{selected_account[-4:]} is ready, {first_name}. How can I help you today?")
                        else:
                            # Localized fallback
                            if user_language == "roman_urdu":
                                return f"Perfect! Account ***-***-{selected_account[-4:]} ready hai, {first_name}. Main aapki kaise madad kar sakta hoon?"
                            elif user_language == "urdu_script":
                                return f"بہترین! اکاؤنٹ ***-***-{selected_account[-4:]} تیار ہے، {first_name}۔ میں آپ کی کیسے مدد کر سکتا ہوں؟"
                            else:
                                return f"Perfect! Account ***-***-{selected_account[-4:]} is ready, {first_name}. How can I help you today?"
                    
                    else:
                        return await api_call_error_response("Account verification failed", user_message, first_name, user_language)
                
                else:
                    # Return the selection guidance from API
                    return selection_result.get("response", get_localized_response("account_selection_help", user_language))
            
            else:
                return await api_call_error_response("Account selection API failed", user_message, first_name, user_language)
        
    except Exception as e:
        logger.error({
            "action": "account_selection_api_error",
            "sender_id": sender_id,
            "user_language": user_language,
            "error": str(e)
        })
        return await api_call_error_response(str(e), user_message, first_name, user_language)

async def handle_authenticated_queries_api_based(sender_id: str, user_message: str, user_language: str) -> str:
    """Handle authenticated user queries with pure API calls and language support."""
    
    user_info = get_user_account_info(sender_id)
    account_number = user_info["account_number"]
    first_name = user_info["name"].split()[0]
    
    try:
        logger.info({
            "action": "processing_authenticated_query_api",
            "sender_id": sender_id,
            "user_language": user_language,
            "account_number": account_number,
            "user_message": user_message[:100] + "..." if len(user_message) > 100 else user_message,
            "message_length": len(user_message)
        })
        
        # API call for main query processing
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/process_query",
                json={
                    "user_message": user_message,
                    "account_number": account_number,
                    "first_name": first_name,
                    "user_language": user_language
                }
            )
            
            # Log response details for debugging
            logger.info({
                "action": "backend_api_response",
                "status_code": response.status_code,
                "response_size": len(response.text),
                "user_language": user_language
            })
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if the API response indicates an error
                if result.get("status") == "error":
                    logger.error({
                        "action": "backend_api_returned_error",
                        "sender_id": sender_id,
                        "user_language": user_language,
                        "api_error": result.get("error", "Unknown error"),
                        "api_response": result.get("response", "No response")
                    })
                    return result.get("response", get_localized_response("technical_error", user_language, name=f", {first_name}"))
                
                logger.info({
                    "action": "authenticated_query_processed_successfully",
                    "sender_id": sender_id,
                    "user_language": user_language,
                    "response_length": len(result.get("response", ""))
                })
                
                return result.get("response", "I processed your request but couldn't generate a response.")
            
            else:
                logger.error({
                    "action": "backend_api_http_error",
                    "sender_id": sender_id,
                    "user_language": user_language,
                    "status_code": response.status_code,
                    "response_text": response.text[:500]
                })
                return await api_call_error_response(f"Backend API error: {response.status_code}", user_message, first_name, user_language)
        
    except httpx.TimeoutException as e:
        logger.error({
            "action": "backend_api_timeout",
            "sender_id": sender_id,
            "user_language": user_language,
            "error": str(e)
        })
        return await api_call_error_response("Request timeout", user_message, first_name, user_language)
        
    except httpx.RequestError as e:
        logger.error({
            "action": "backend_api_connection_error",
            "sender_id": sender_id,
            "user_language": user_language,
            "error": str(e)
        })
        return await api_call_error_response("Connection error", user_message, first_name, user_language)
        
    except Exception as e:
        logger.error({
            "action": "authenticated_query_api_error",
            "sender_id": sender_id,
            "user_language": user_language,
            "error": str(e),
            "user_message": user_message,
            "error_type": type(e).__name__
        })
        return await api_call_error_response(str(e), user_message, first_name, user_language)

# Helper API call functions with language support
async def api_call_initial_greeting(user_language: str = "english") -> str:
    """API call for initial greeting with language support."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/handle_initial_greeting",
                json={"user_language": user_language}
            )
            if response.status_code == 200:
                return response.json().get("response", get_localized_response("initial_choices", user_language))
    except Exception as e:
        logger.error(f"Initial greeting API call failed: {e}")
    
    return get_localized_response("initial_choices", user_language)

async def api_call_extract_cnic(text: str) -> str:
    """API call for CNIC extraction."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/extract_cnic_from_text",
                json={"text": text}
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("cnic", None)
    except Exception as e:
        logger.error(f"CNIC extraction API call failed: {e}")
    
    return None

async def api_call_handle_invalid_cnic(user_input: str, user_language: str) -> str:
    """API call for invalid CNIC handling with language support."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/handle_invalid_cnic_format",
                json={
                    "user_input": user_input,
                    "first_name": "",
                    "user_language": user_language
                }
            )
            if response.status_code == 200:
                return response.json().get("response", get_localized_response("cnic_format_help", user_language))
    except Exception as e:
        logger.error(f"Invalid CNIC API call failed: {e}")
    
    return get_localized_response("cnic_format_help", user_language)

async def api_call_cnic_verification_success(user_data: dict, user_language: str) -> str:
    """API call for CNIC verification success with language support."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/handle_cnic_verification_success",
                json={
                    "user_name": user_data["name"],
                    "accounts": user_data["accounts"],
                    "cnic": user_data["cnic"],
                    "user_language": user_language
                }
            )
            if response.status_code == 200:
                return response.json().get("response", "CNIC verified! Please select your account.")
    except Exception as e:
        logger.error(f"CNIC verification success API call failed: {e}")
    
    first_name = user_data["name"].split()[0]
    if user_language == "roman_urdu":
        return f"Swagat hai {first_name}! Please apna account select kariye."
    elif user_language == "urdu_script":
        return f"خیر مقدم {first_name}! براہ کرم اپنا اکاؤنٹ منتخب کریں۔"
    else:
        return f"Welcome {first_name}! Please select your account."

async def api_call_cnic_verification_failure(cnic: str, user_language: str) -> str:
    """API call for CNIC verification failure with language support."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/handle_cnic_verification_failure",
                json={
                    "cnic": cnic,
                    "first_name": "",
                    "user_language": user_language
                }
            )
            if response.status_code == 200:
                return response.json().get("response", get_localized_response("cnic_not_found", user_language))
    except Exception as e:
        logger.error(f"CNIC verification failure API call failed: {e}")
    
    return get_localized_response("cnic_not_found", user_language)

async def api_call_error_response(error: str, user_message: str, first_name: str, user_language: str) -> str:
    """API call for error response generation with language support."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/generate_error_response",
                json={
                    "error": error,
                    "user_message": user_message,
                    "first_name": first_name,
                    "user_language": user_language
                }
            )
            if response.status_code == 200:
                return response.json().get("response", get_localized_response("technical_error", user_language, name=f", {first_name}" if first_name else ""))
    except Exception as e:
        logger.error(f"Error response API call failed: {e}")
    
    return get_localized_response("technical_error", user_language, name=f", {first_name}" if first_name else "")

def send_message(recipient_id, message_text):
    """Send response to Facebook Messenger with enhanced error handling and message splitting."""
    
    # Facebook Messenger has a limit of ~2000 characters per message
    MAX_MESSAGE_LENGTH = 1900  # Leave some buffer
    
    if len(message_text) <= MAX_MESSAGE_LENGTH:
        # Send single message
        _send_single_message(recipient_id, message_text)
    else:
        # Split long message into multiple parts
        _send_long_message_in_parts(recipient_id, message_text, MAX_MESSAGE_LENGTH)

def _send_single_message(recipient_id, message_text):
    """Send a single message to Facebook Messenger."""
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
            "response_status": response.status_code,
            "message_length": len(message_text)
        })
    except requests.exceptions.RequestException as e:
        logger.error({
            "action": "send_message_error",
            "recipient_id": recipient_id,
            "error": str(e),
            "message_length": len(message_text)
        })

def _send_long_message_in_parts(recipient_id, message_text, max_length):
    """Split and send long messages in multiple parts."""
    try:
        # Try to split intelligently by sections first
        if '\n\n' in message_text:
            # Split by double newlines (paragraphs)
            sections = message_text.split('\n\n')
            current_part = ""
            
            for section in sections:
                if len(current_part) + len(section) + 2 > max_length:
                    if current_part.strip():
                        _send_single_message(recipient_id, current_part.strip())
                        current_part = ""
                        time.sleep(0.3)  # Brief delay between parts
                    
                    # If single section is too long, split it further
                    if len(section) > max_length:
                        _split_section_by_lines(recipient_id, section, max_length)
                    else:
                        current_part = section + "\n\n"
                else:
                    current_part += section + "\n\n"
            
            # Send remaining part
            if current_part.strip():
                _send_single_message(recipient_id, current_part.strip())
        
        else:
            # Fallback to line-by-line splitting
            _split_section_by_lines(recipient_id, message_text, max_length)
        
        logger.info({
            "action": "long_message_sent_successfully",
            "recipient_id": recipient_id,
            "original_length": len(message_text)
        })
        
    except Exception as e:
        logger.error({
            "action": "send_long_message_error", 
            "recipient_id": recipient_id,
            "error": str(e),
            "message_length": len(message_text)
        })
        
        # Fallback: send truncated message
        truncated_message = message_text[:max_length-100] + "\n\n[Message truncated due to length. Please ask for specific parts if needed.]"
        _send_single_message(recipient_id, truncated_message)

def _split_section_by_lines(recipient_id, section_text, max_length):
    """Split a section by lines when it's too long."""
    lines = section_text.split('\n')
    current_part = ""
    
    for line in lines:
        if len(current_part) + len(line) + 1 > max_length:
            if current_part.strip():
                _send_single_message(recipient_id, current_part.strip())
                current_part = ""
                time.sleep(0.3)
            
            # If single line is too long, split by words
            if len(line) > max_length:
                words = line.split(' ')
                temp_line = ""
                for word in words:
                    if len(temp_line) + len(word) + 1 <= max_length:
                        temp_line += word + " "
                    else:
                        if temp_line.strip():
                            current_part += temp_line.strip() + "\n"
                        temp_line = word + " "
                current_part += temp_line.strip() + "\n"
            else:
                current_part += line + "\n"
        else:
            current_part += line + "\n"
    
    # Send remaining part
    if current_part.strip():
        _send_single_message(recipient_id, current_part.strip())

def send_typing_indicator(recipient_id):
    """Send typing indicator to show processing."""
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": recipient_id},
        "sender_action": "typing_on"
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error({
            "action": "send_typing_indicator_error",
            "recipient_id": recipient_id,
            "error": str(e)
        })

@app.get("/webhook_health")
async def webhook_health_check():
    """Webhook health check with backend connectivity test."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            backend_healthy = response.status_code == 200
            backend_response = response.json() if backend_healthy else {}
    except Exception as e:
        backend_healthy = False
        backend_response = {"error": str(e)}
    
    return {
        "status": "healthy",
        "webhook_service": "operational",
        "backend_connection": "healthy" if backend_healthy else "unhealthy",
        "backend_response": backend_response,
        "processed_messages": len(processed_messages),
        "authenticated_users": len(authenticated_users),
        "timestamp": datetime.now().isoformat(),
        "service": "pure_api_based_banking_webhook",
        "architecture": "webhook_makes_api_calls_only",
        "ai_agent_location": "backend_only",
        "communication_method": "http_api_calls_only",
        "message_splitting": "enabled_for_facebook_limits",
        "language_support": "english_roman_urdu_urdu_script"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    return await webhook_health_check()

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Pure API-Based Banking Webhook with Language Support...")
    logger.info("📡 Architecture: Webhook → HTTP API → Backend AI Agent")
    logger.info("🔄 Communication: Pure API calls only, no direct imports")
    logger.info("🎯 Separation: Clean microservices architecture")
    logger.info("📱 Message Splitting: Enabled for Facebook Messenger limits")
    logger.info("🌐 Language Support: English, Roman Urdu, Urdu Script")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)