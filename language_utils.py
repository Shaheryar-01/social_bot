# language_utils.py - Language Detection and Localized Responses
import re
from typing import Dict, Any

def detect_user_language(user_message: str) -> str:
    """Detect if user is using English, Urdu script, or Roman Urdu"""
    if not user_message or not user_message.strip():
        return "english"
    
    # Urdu script detection (Arabic/Urdu Unicode range)
    if re.search(r'[\u0600-\u06FF]', user_message):
        return "urdu_script"
    
    # Roman Urdu keywords detection
    roman_urdu_words = [
        # Common Urdu words in Roman script
        "mujhe", "mere", "mera", "meri", "kya", "hai", "hain", "kar", "karo", 
        "dikhao", "dikhaiye", "batao", "bataiye", "kitna", "kitni", "kahan", 
        "kaise", "kyun", "kyu", "aap", "app", "se", "mein", "main", "ki", "ka", 
        "ke", "ko", "par", "pe", "wala", "wali", "walay", "transaction", "paisa", 
        "paise", "rupay", "rupaiye", "account", "balance", "check", "dekho", 
        "dekhiye", "bhejo", "transfer", "send", "spending", "kharcha", "kharch",
        "sab", "sabse", "zyada", "ziyada", "kam", "mehenga", "mehenga", "sasta",
        "total", "sum", "average", "comparison", "compare", "mukabla", "tulna",
        "bank", "banking", "service", "services", "madad", "help", "saath",
        "malik", "owner", "customer", "grahak", "time", "waqt", "date", "tarikh",
        "month", "mahina", "year", "saal", "din", "day", "raat", "night",
        "subah", "morning", "sham", "evening", "abhi", "now", "phir", "again",
        "dobara", "wapis", "back", "return", "jana", "go", "aana", "come",
        "lena", "take", "dena", "give", "receive", "hasil", "mila", "got"
    ]
    
    message_lower = user_message.lower()
    total_words = len(user_message.split())
    
    if total_words == 0:
        return "english"
    
    # Count Roman Urdu words
    roman_urdu_count = sum(1 for word in roman_urdu_words if word in message_lower)
    
    # If more than 25% of words are Roman Urdu, classify as Roman Urdu
    if roman_urdu_count / total_words >= 0.25:
        return "roman_urdu"
    
    # Check for Roman Urdu patterns
    roman_urdu_patterns = [
        r'\b(mujhe|mere|mera)\b',
        r'\b(dikhao|batao|karo)\b',
        r'\b(kitna|kahan|kaise)\b',
        r'\b(paisa|rupay|account)\b',
        r'\b(sab\s*se|zyada|kam)\b'
    ]
    
    for pattern in roman_urdu_patterns:
        if re.search(pattern, message_lower):
            return "roman_urdu"
    
    return "english"

def get_language_instruction(user_language: str) -> str:
    """Get language instruction for LLM based on detected language"""
    instructions = {
        "english": "RESPOND IN ENGLISH. Use natural, conversational English and match the user's formal/casual tone.",
        "roman_urdu": "RESPOND IN ROMAN URDU (Urdu written in English alphabet like 'aap', 'mujhe', 'dikhao'). Use the same Roman Urdu style as the user. Mix with English banking terms where appropriate.",
        "urdu_script": "RESPOND IN URDU SCRIPT (اردو). Use proper Urdu script and match the user's formal/informal tone."
    }
    return instructions.get(user_language, instructions["english"])

def get_localized_response(message_key: str, user_language: str, **params) -> str:
    """Get localized hardcoded responses for common messages"""
    
    responses = {
        "rate_limit": {
            "english": "Please wait a moment before sending another message. 😊",
            "roman_urdu": "Thoda intezar kariye pehle message bhejne se. 😊",
            "urdu_script": "براہ کرم پیغام بھیجنے سے پہلے تھوڑا انتظار کریں۔ 😊"
        },
        "session_ended": {
            "english": "Session ended. Goodbye!",
            "roman_urdu": "Session khatam. Allah hafiz!",
            "urdu_script": "سیشن ختم۔ اللہ حافظ!"
        },
        "processing": {
            "english": "🤔 Processing your request...",
            "roman_urdu": "🤔 Aapka request process kar raha hoon...",
            "urdu_script": "🤔 آپ کی درخواست پر کام کر رہا ہوں..."
        },
        "account_access_start": {
            "english": "Perfect! I'll help you access your account. Please provide your CNIC in the format 12345-1234567-1 to get started.",
            "roman_urdu": "Perfect! Main aapke account access mein madad karoonga. Please apna CNIC 12345-1234567-1 format mein dijiye.",
            "urdu_script": "بہترین! میں آپ کے اکاؤنٹ تک رسائی میں مدد کروں گا۔ براہ کرم اپنا شناختی کارڈ 12345-1234567-1 فارمیٹ میں دیں۔"
        },
        "cnic_format_help": {
            "english": "Please provide your CNIC in the format 12345-1234567-1 to continue.",
            "roman_urdu": "Please apna CNIC 12345-1234567-1 format mein dijiye continue karne ke liye.",
            "urdu_script": "براہ کرم اپنا شناختی کارڈ 12345-1234567-1 فارمیٹ میں دیں۔"
        },
        "cnic_not_found": {
            "english": "CNIC not found in our system. Please check the format and try again.",
            "roman_urdu": "CNIC hamare system mein nahi mila. Please format check karke dobara try kariye.",
            "urdu_script": "شناختی کارڈ ہمارے سسٹم میں نہیں ملا۔ براہ کرم فارمیٹ چیک کر کے دوبارہ کوشش کریں۔"
        },
        "account_selection_help": {
            "english": "Please select your account using the last 4 digits or specify 'USD account' or 'PKR account'.",
            "roman_urdu": "Please apna account last 4 digits se select kariye ya 'USD account' ya 'PKR account' kahiye.",
            "urdu_script": "براہ کرم اپنا اکاؤنٹ آخری 4 ہندسوں سے منتخب کریں یا 'USD account' یا 'PKR account' کہیں۔"
        },
        "technical_error": {
            "english": "I'm experiencing technical difficulties{name}. Please try again in a moment.",
            "roman_urdu": "Technical problem aa rahi hai{name}. Thoda baad try kariye please.",
            "urdu_script": "تکنیکی مسئلہ ہو رہا ہے{name}۔ براہ کرم تھوڑی دیر بعد کوشش کریں۔"
        },
        "initial_choices": {
            "english": "Welcome to Best Bank! I can help you with:\n1. General bank information\n2. Personal account access\n\nWhich would you like?",
            "roman_urdu": "Best Bank mein aapka swagat hai! Main aapki madad kar sakta hoon:\n1. Bank ki general information\n2. Personal account access\n\nKya chahiye aapko?",
            "urdu_script": "بیسٹ بینک میں آپ کا خیر مقدم! میں آپ کی مدد کر سکتا ہوں:\n1. بینک کی عمومی معلومات\n2. ذاتی اکاؤنٹ تک رسائی\n\nآپ کو کیا چاہیے؟"
        },
        "bank_info_help": {
            "english": "I can provide information about Best Bank's services, hours, and policies. What would you like to know?",
            "roman_urdu": "Main aapko Best Bank ki services, hours, aur policies ke baare mein bata sakta hoon. Kya jaanna chahte hain?",
            "urdu_script": "میں آپ کو بیسٹ بینک کی خدمات، اوقات اور پالیسیوں کے بارے میں بتا سکتا ہوں۔ کیا جاننا چاہتے ہیں؟"
        }
    }
    
    # Get the response for the language, fallback to English
    response_dict = responses.get(message_key, {})
    response = response_dict.get(user_language, response_dict.get("english", ""))
    
    # Handle parameters like {name}
    if params:
        for key, value in params.items():
            placeholder = "{" + key + "}"
            if placeholder in response:
                if key == "name" and value:
                    # Add space before name for proper formatting
                    name_part = f" {value}" if user_language == "english" else f" {value}"
                    response = response.replace(placeholder, name_part)
                else:
                    response = response.replace(placeholder, str(value) if value else "")
    
    return response

def get_language_aware_prompt_prefix(user_message: str) -> str:
    """Get language-aware prompt prefix for any LLM call"""
    user_language = detect_user_language(user_message)
    language_instruction = get_language_instruction(user_language)
    
    return f"""CRITICAL LANGUAGE INSTRUCTION: {language_instruction}

DETECTED USER LANGUAGE: {user_language.upper()}
USER'S MESSAGE: "{user_message}"

"""

# Export main functions
__all__ = [
    'detect_user_language', 
    'get_language_instruction', 
    'get_localized_response',
    'get_language_aware_prompt_prefix'
]