from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mongo import transactions
from typing import Dict, Any, List, Optional
from datetime import datetime
from bson import ObjectId
import json
import re
import logging
from ai_agent import enhanced_ai_agent
from language_utils import detect_user_language, get_localized_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class CNICVerifyRequest(BaseModel):
    cnic: str
    user_language: Optional[str] = "english"

class AccountSelectionRequest(BaseModel):
    cnic: str
    account_number: str
    user_language: Optional[str] = "english"

class UserBalanceQuery(BaseModel):
    account_number: str
    user_language: Optional[str] = "english"

class MoneyTransferRequest(BaseModel):
    from_account: str
    to_recipient: str
    amount: float
    currency: str = "PKR"
    user_language: Optional[str] = "english"

class PipelineQuery(BaseModel):
    account_number: str
    pipeline: List[Dict[str, Any]]
    user_language: Optional[str] = "english"

class ProcessQueryRequest(BaseModel):
    user_message: str
    account_number: str
    first_name: str
    user_language: Optional[str] = None

class ProcessQueryResponse(BaseModel):
    status: str
    response: str
    error: Optional[str] = None

class RAGQueryRequest(BaseModel):
    user_message: str
    first_name: str
    user_language: Optional[str] = None

class RAGQueryResponse(BaseModel):
    status: str
    response: str
    context_used: List[str] = []
    error: Optional[str] = None

class EnhancedQueryRequest(BaseModel):
    user_message: str
    account_number: str
    first_name: str
    query_type: Optional[str] = None
    user_language: Optional[str] = None

class EnhancedQueryResponse(BaseModel):
    status: str
    response: str
    query_classification: Optional[Dict] = None
    processing_method: Optional[str] = None
    context_stored: Optional[bool] = None
    error: Optional[str] = None

# NEW API-BASED COMMUNICATION MODELS WITH LANGUAGE SUPPORT
class InitialChoiceRequest(BaseModel):
    user_message: str
    first_name: str
    user_language: Optional[str] = None

class InitialChoiceResponse(BaseModel):
    status: str
    choice_detected: str
    confidence: str
    reasoning: str
    response: Optional[str] = None

class ExtractCNICRequest(BaseModel):
    text: str

class ExtractCNICResponse(BaseModel):
    status: str
    cnic: Optional[str] = None
    found: bool

class InvalidCNICRequest(BaseModel):
    user_input: str
    first_name: str
    user_language: Optional[str] = "english"

class CNICVerificationSuccessRequest(BaseModel):
    user_name: str
    accounts: List[str]
    cnic: str
    user_language: Optional[str] = "english"

class CNICVerificationFailureRequest(BaseModel):
    cnic: str
    first_name: str
    user_language: Optional[str] = "english"

class EnhancedAccountSelectionRequest(BaseModel):
    user_input: str
    available_accounts: List[str]
    first_name: str
    user_language: Optional[str] = "english"

class AccountConfirmationRequest(BaseModel):
    selected_account: str
    user_name: str
    user_language: Optional[str] = "english"

class SessionEndRequest(BaseModel):
    account_number: str
    first_name: str
    user_language: Optional[str] = "english"

class ErrorResponseRequest(BaseModel):
    error: str
    user_message: str
    first_name: str
    user_language: Optional[str] = "english"

class GenericResponse(BaseModel):
    status: str
    response: str
    error: Optional[str] = None

def convert_objectid_to_string(doc):
    """Recursively convert ObjectId to string in documents."""
    if isinstance(doc, dict):
        return {k: convert_objectid_to_string(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [convert_objectid_to_string(item) for item in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc

def process_pipeline_dates(pipeline):
    """Process pipeline to handle date objects properly."""
    processed_pipeline = []
    for stage in pipeline:
        processed_stage = {}
        for key, value in stage.items():
            if isinstance(value, dict):
                processed_stage[key] = process_dict_dates(value)
            else:
                processed_stage[key] = value
        processed_pipeline.append(processed_stage)
    return processed_pipeline

def process_dict_dates(obj):
    """Recursively process dictionary to handle date objects."""
    if isinstance(obj, dict):
        processed = {}
        for k, v in obj.items():
            if isinstance(v, dict):
                if "$date" in v:
                    try:
                        processed[k] = datetime.fromisoformat(v["$date"].replace("Z", "+00:00"))
                    except ValueError as e:
                        logger.error({
                            "action": "process_dict_dates",
                            "error": f"Invalid date format: {v['$date']}"
                        })
                        processed[k] = v
                elif "$gte" in v or "$lte" in v or "$lt" in v or "$gt" in v:
                    processed[k] = {}
                    for op, date_val in v.items():
                        if isinstance(date_val, dict) and "$date" in date_val:
                            try:
                                processed[k][op] = datetime.fromisoformat(date_val["$date"].replace("Z", "+00:00"))
                            except ValueError as e:
                                logger.error({
                                    "action": "process_dict_dates",
                                    "error": f"Invalid date format: {date_val['$date']}"
                                })
                                processed[k][op] = date_val
                        else:
                            processed[k][op] = date_val
                else:
                    processed[k] = process_dict_dates(v)
            elif isinstance(v, list):
                processed[k] = [process_dict_dates(item) for item in v]
            else:
                processed[k] = v
        return processed
    else:
        return obj

@router.post("/verify_cnic")
def verify_cnic(data: CNICVerifyRequest):
    """Verify user by CNIC and return available accounts with language support."""
    try:
        # Clean and format CNIC
        cnic = data.cnic.strip()
        user_language = data.user_language or "english"
        
        logger.info(f"🔍 Verifying CNIC: {cnic} (Language: {user_language})")
        
        # Find user by CNIC in transactions collection
        user_record = transactions.find_one({"cnic": cnic})
        
        if user_record:
            # Get all unique accounts for this CNIC
            accounts = list(transactions.distinct("account_number", {"cnic": cnic}))
            
            logger.info(f"✅ CNIC verified. Found {len(accounts)} accounts for {user_record['name']}")
            
            return {
                "status": "success",
                "user": {
                    "name": user_record["name"],
                    "cnic": cnic,
                    "accounts": accounts
                },
                "user_language": user_language
            }
        else:
            logger.warning(f"❌ CNIC not found: {cnic}")
            return {"status": "fail", "reason": "CNIC not found", "user_language": user_language}
            
    except Exception as e:
        logger.error(f"CNIC verification error: {e}")
        return {"status": "fail", "reason": "Verification failed", "user_language": user_language}

@router.post("/select_account")
def select_account(data: AccountSelectionRequest):
    """Confirm account selection for verified CNIC with language support."""
    try:
        cnic = data.cnic.strip()
        account_number = data.account_number.strip()
        user_language = data.user_language or "english"
        
        logger.info(f"🔍 Verifying account selection: CNIC {cnic}, Account {account_number} (Language: {user_language})")
        
        # Verify that this account belongs to this CNIC
        account_record = transactions.find_one({
            "cnic": cnic,
            "account_number": account_number
        })
        
        if account_record:
            logger.info(f"✅ Account selection verified for {account_record['name']}")
            
            return {
                "status": "success",
                "user": {
                    "name": account_record["name"],
                    "cnic": cnic,
                    "selected_account": account_number
                },
                "user_language": user_language
            }
        else:
            logger.warning(f"❌ Account {account_number} not found for CNIC {cnic}")
            return {"status": "fail", "reason": "Account not found for this CNIC", "user_language": user_language}
            
    except Exception as e:
        logger.error(f"Account selection error: {e}")
        return {"status": "fail", "reason": "Account selection failed", "user_language": user_language}

@router.post("/user_balance")
async def get_user_balance(data: UserBalanceQuery):
    """Get user's current balance for selected account with language support."""
    try:
        account_number = data.account_number.strip()
        user_language = data.user_language or "english"
        
        # Get user info from latest transaction
        user_record = transactions.find_one({"account_number": account_number})
        if not user_record:
            return {"status": "fail", "reason": "Account not found", "user_language": user_language}
        
        # Get latest transaction for most current balance
        latest_txn = transactions.find_one(
            {"account_number": account_number},
            sort=[("date", -1), ("_id", -1)]
        )
        
        if latest_txn:
            # Extract balance from account_balance field
            current_balance = latest_txn.get("account_balance", 0)
            account_currency = latest_txn.get("account_currency", "pkr")
        else:
            current_balance = 0
            account_currency = "pkr"
        
        # Format balance by currency
        if account_currency.lower() == "usd":
            balance_usd = current_balance
            balance_pkr = 0
        else:
            balance_usd = 0 
            balance_pkr = current_balance
        
        return {
            "status": "success",
            "user": {
                "first_name": user_record["name"].split()[0],
                "last_name": user_record["name"].split()[-1] if len(user_record["name"].split()) > 1 else "",
                "account_number": account_number,
                "current_balance_usd": balance_usd,
                "current_balance_pkr": balance_pkr,
                "account_currency": account_currency
            },
            "user_language": user_language
        }
    except Exception as e:
        logger.error(f"Balance error: {e}")
        return {"status": "fail", "error": str(e), "user_language": user_language}

# DEBUG ENDPOINT TO CHECK RESPONSE LENGTHS WITH LANGUAGE SUPPORT
@router.post("/debug_query_length")
async def debug_query_length(data: ProcessQueryRequest):
    """Debug endpoint to check response length without sending to messenger."""
    try:
        # Auto-detect language if not provided
        user_language = data.user_language or detect_user_language(data.user_message)
        
        # Process the query
        response = await enhanced_ai_agent.process_query(
            user_message=data.user_message,
            account_number=data.account_number,
            first_name=data.first_name
        )
        
        return {
            "status": "success",
            "response_length": len(response),
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "full_response": response,
            "facebook_limit": 2000,
            "exceeds_limit": len(response) > 1900,
            "user_language": user_language
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "user_language": data.user_language or "english"
        }

@router.post("/execute_pipeline")
async def execute_pipeline(data: PipelineQuery):
    """Execute a dynamic MongoDB aggregation pipeline with language support."""
    try:
        user_language = data.user_language or "english"
        
        # Validate input
        if not data.pipeline:
            return {"status": "fail", "reason": "Empty pipeline provided", "user_language": user_language}
        
        if not data.account_number:
            return {"status": "fail", "reason": "Account number is required", "user_language": user_language}
        
        # Process pipeline to handle date objects
        processed_pipeline = process_pipeline_dates(data.pipeline)
        
        logger.info(f"Executing pipeline for account {data.account_number}: {processed_pipeline}")
        
        # Execute pipeline on transactions collection
        result = list(transactions.aggregate(processed_pipeline))
        
        # Convert ObjectId to string for JSON serialization
        result = convert_objectid_to_string(result)
        
        logger.info(f"Pipeline execution successful. Returned {len(result)} documents")
        
        return {
            "status": "success",
            "data": result,
            "count": len(result),
            "user_language": user_language
        }
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")
        return {"status": "fail", "error": str(e), "user_language": user_language}

@router.post("/transfer_money")
async def transfer_money(data: MoneyTransferRequest):
    """Handle money transfer with enhanced validation and language support."""
    try:
        user_language = data.user_language or "english"
        
        # Validate input
        if data.amount <= 0:
            return {"status": "fail", "reason": "Transfer amount must be positive", "user_language": user_language}
        
        if data.currency.upper() not in ["USD", "PKR"]:
            return {"status": "fail", "reason": "Currency must be USD or PKR", "user_language": user_language}
        
        # Get account info
        account_record = transactions.find_one({"account_number": data.from_account})
        if not account_record:
            return {"status": "fail", "reason": "Sender account not found", "user_language": user_language}
        
        # Get current balance from latest transaction
        latest_txn = transactions.find_one(
            {"account_number": data.from_account},
            sort=[("date", -1), ("_id", -1)]
        )
        
        if latest_txn:
            current_balance = latest_txn.get("account_balance", 0)
            account_currency = latest_txn.get("account_currency", "pkr")
        else:
            return {"status": "fail", "reason": "No transaction history found", "user_language": user_language}
        
        # Check if currencies match
        if data.currency.upper() != account_currency.upper():
            return {
                "status": "fail", 
                "reason": f"Cannot transfer {data.currency} from {account_currency} account",
                "user_language": user_language
            }
        
        # Check sufficient balance
        if current_balance < data.amount:
            return {
                "status": "fail", 
                "reason": f"Insufficient {data.currency} balance. Available: {current_balance:.2f}, Required: {data.amount:.2f}",
                "user_language": user_language
            }
        
        # Calculate new balance
        new_balance = current_balance - data.amount
        
        # Create transfer transaction
        transfer_txn = {
            "name": account_record["name"],
            "cnic": account_record["cnic"],
            "account_number": data.from_account,
            "date": datetime.now(),
            "type": "debit",
            "description": f"Transfer to {data.to_recipient}",
            "category": "Transfer",
            "account_currency": account_currency.lower(),
            "amount_deducted_from_account": data.amount,
            "transaction_amount": data.amount,
            "transaction_currency": data.currency.lower(),
            "account_balance": new_balance
        }
        
        # Insert transaction
        txn_result = transactions.insert_one(transfer_txn)
        
        logger.info(f"Transfer successful: {data.amount} {data.currency} from {data.from_account} to {data.to_recipient}")
        logger.info(f"Updated balance: {new_balance} {account_currency}")
        
        return {
            "status": "success",
            "message": f"Successfully transferred {data.amount} {data.currency} to {data.to_recipient}",
            "transaction_id": str(txn_result.inserted_id),
            "new_balance": new_balance,
            "currency": account_currency,
            "transfer_details": {
                "amount": data.amount,
                "currency": data.currency,
                "recipient": data.to_recipient,
                "timestamp": transfer_txn["date"].isoformat()
            },
            "user_language": user_language
        }
    except Exception as e:
        logger.error(f"Transfer error: {e}")
        return {"status": "fail", "error": str(e), "user_language": user_language}

@router.post("/process_query", response_model=ProcessQueryResponse)
async def process_query(data: ProcessQueryRequest):
    """Process user banking queries using enhanced hybrid AI agent with language support."""
    try:
        # Auto-detect language if not provided
        user_language = data.user_language or detect_user_language(data.user_message)
        
        logger.info({
            "action": "api_process_query_start",
            "user_message": data.user_message,
            "account_number": data.account_number,
            "first_name": data.first_name,
            "user_language": user_language
        })
        
        # Check if this is a bank information query that should use RAG
        bank_info_keywords = [
            "tell me about best bank", "tell me about the bank", "about best bank", 
            "bank information", "bank services", "what services", "bank hours",
            "about the bank", "bank policies", "loan information", "credit card info",
            # Roman Urdu keywords
            "bank ke baare mein", "bank ki information", "bank ki services", 
            "best bank kya hai", "bank ke hours", "bank ki policies"
        ]
        
        user_message_lower = data.user_message.lower().strip()
        is_bank_info_query = any(keyword in user_message_lower for keyword in bank_info_keywords)
        
        if is_bank_info_query:
            logger.info({
                "action": "detected_bank_info_query",
                "user_message": data.user_message,
                "user_language": user_language,
                "routing_to": "rag_system"
            })
            
            # Route to RAG system for bank information
            try:
                rag_response = await enhanced_ai_agent.generate_rag_response(
                    query=data.user_message,
                    first_name=data.first_name
                )
                
                logger.info({
                    "action": "rag_response_generated",
                    "account_number": data.account_number,
                    "user_language": user_language,
                    "response_length": len(rag_response)
                })
                
                return ProcessQueryResponse(
                    status="success",
                    response=rag_response
                )
                
            except Exception as rag_error:
                logger.error(f"RAG processing error: {rag_error}")
                # Fallback to generic bank info response
                fallback_response = await enhanced_ai_agent.response_generator.generate_response(
                    "Bank information query", 
                    {"query": data.user_message, "source": "fallback"}, 
                    data.user_message, 
                    data.first_name, 
                    "", 
                    "rag"
                )
                return ProcessQueryResponse(
                    status="success",
                    response=fallback_response
                )
        
        # Use the enhanced hybrid AI agent to process other queries
        response = await enhanced_ai_agent.process_query(
            user_message=data.user_message,
            account_number=data.account_number,
            first_name=data.first_name
        )
        
        logger.info({
            "action": "api_process_query_success",
            "account_number": data.account_number,
            "user_language": user_language,
            "response_length": len(response)
        })
        
        return ProcessQueryResponse(
            status="success",
            response=response
        )
        
    except Exception as e:
        user_language = data.user_language or detect_user_language(data.user_message)
        logger.error({
            "action": "api_process_query_error",
            "error": str(e),
            "account_number": data.account_number,
            "user_message": data.user_message,
            "user_language": user_language,
            "error_type": type(e).__name__
        })
        
        # Generate a proper error response
        try:
            error_response = await enhanced_ai_agent.response_generator.generate_response(
                "Query processing error", 
                {"error": str(e)}, 
                data.user_message, 
                data.first_name, 
                "", 
                "error"
            )
        except Exception as inner_e:
            logger.error(f"Error generating error response: {inner_e}")
            error_response = get_localized_response("technical_error", user_language, name=f", {data.first_name}")
        
        return ProcessQueryResponse(
            status="error",
            response=error_response,
            error=str(e)
        )

# NEW API ENDPOINTS FOR PURE API-BASED COMMUNICATION WITH LANGUAGE SUPPORT

@router.post("/detect_initial_choice", response_model=InitialChoiceResponse)
async def detect_initial_choice(data: InitialChoiceRequest):
    """Detect user's initial choice for RAG vs Account mode with language support."""
    try:
        # Auto-detect language if not provided
        user_language = data.user_language or detect_user_language(data.user_message)
        
        logger.info({
            "action": "api_detect_initial_choice",
            "user_message": data.user_message,
            "first_name": data.first_name,
            "user_language": user_language
        })
        
        # Use the enhanced AI agent's method
        choice_result = await enhanced_ai_agent.detect_initial_choice(
            data.user_message, data.first_name
        )
        
        return InitialChoiceResponse(
            status="success",
            choice_detected=choice_result.get("choice_detected", "none"),
            confidence=choice_result.get("confidence", "low"),
            reasoning=choice_result.get("reasoning", ""),
            response=choice_result.get("response", "")
        )
        
    except Exception as e:
        user_language = data.user_language or "english"
        logger.error(f"Initial choice detection API error: {e}")
        return InitialChoiceResponse(
            status="error",
            choice_detected="none",
            confidence="low",
            reasoning=f"Error: {e}",
            response=get_localized_response("initial_choices", user_language)
        )

@router.post("/handle_initial_greeting", response_model=GenericResponse)
async def handle_initial_greeting(request: dict = None):
    """Handle initial greeting from user with language support."""
    try:
        user_language = "english"
        if request and "user_language" in request:
            user_language = request["user_language"]
        
        response = await enhanced_ai_agent.handle_initial_greeting()
        
        return GenericResponse(
            status="success",
            response=response
        )
        
    except Exception as e:
        logger.error(f"Initial greeting API error: {e}")
        user_language = "english"
        if request and "user_language" in request:
            user_language = request["user_language"]
        
        return GenericResponse(
            status="error",
            response=get_localized_response("initial_choices", user_language),
            error=str(e)
        )

@router.post("/extract_cnic_from_text", response_model=ExtractCNICResponse)
async def extract_cnic_from_text(data: ExtractCNICRequest):
    """Extract CNIC from mixed text input."""
    try:
        from ai_agent import extract_cnic_from_text
        
        cnic = extract_cnic_from_text(data.text)
        
        return ExtractCNICResponse(
            status="success",
            cnic=cnic,
            found=cnic is not None
        )
        
    except Exception as e:
        logger.error(f"CNIC extraction API error: {e}")
        return ExtractCNICResponse(
            status="error",
            cnic=None,
            found=False
        )

@router.post("/handle_invalid_cnic_format", response_model=GenericResponse)
async def handle_invalid_cnic_format(data: InvalidCNICRequest):
    """Handle invalid CNIC format with guidance and language support."""
    try:
        user_language = data.user_language or "english"
        
        # Use response generator directly for invalid CNIC format
        response = await enhanced_ai_agent.response_generator.generate_response(
            "Invalid CNIC format provided",
            {"user_input": data.user_input, "format_required": "12345-1234567-1"},
            data.user_input,
            data.first_name,
            "",
            "cnic_validation_error"
        )
        
        return GenericResponse(
            status="success",
            response=response
        )
        
    except Exception as e:
        user_language = data.user_language or "english"
        logger.error(f"Invalid CNIC format API error: {e}")
        return GenericResponse(
            status="error",
            response=get_localized_response("cnic_format_help", user_language),
            error=str(e)
        )

@router.post("/handle_cnic_verification_success", response_model=GenericResponse)
async def handle_cnic_verification_success(data: CNICVerificationSuccessRequest):
    """Handle successful CNIC verification with language support."""
    try:
        user_language = data.user_language or "english"
        
        # Use response generator for CNIC verification success
        response = await enhanced_ai_agent.response_generator.generate_cnic_response(
            success=True,
            user_data={
                "name": data.user_name,
                "accounts": data.accounts,
                "cnic": data.cnic
            },
            user_message=""  # Add this to support language detection
        )
        
        return GenericResponse(
            status="success",
            response=response
        )
        
    except Exception as e:
        user_language = data.user_language or "english"
        logger.error(f"CNIC verification success API error: {e}")
        first_name = data.user_name.split()[0] if data.user_name else ""
        
        if user_language == "roman_urdu":
            fallback = f"Swagat hai {first_name}! Please apna account select kariye."
        elif user_language == "urdu_script":
            fallback = f"خیر مقدم {first_name}! براہ کرم اپنا اکاؤنٹ منتخب کریں۔"
        else:
            fallback = f"Welcome {first_name}! Please select your account."
        
        return GenericResponse(
            status="error",
            response=fallback,
            error=str(e)
        )

@router.post("/handle_cnic_verification_failure", response_model=GenericResponse)
async def handle_cnic_verification_failure(data: CNICVerificationFailureRequest):
    """Handle failed CNIC verification with language support."""
    try:
        user_language = data.user_language or "english"
        
        # Use response generator for CNIC verification failure
        response = await enhanced_ai_agent.response_generator.generate_cnic_response(
            success=False,
            cnic=data.cnic,
            first_name=data.first_name,
            error_reason="CNIC not found",
            user_message=""  # Add this to support language detection
        )
        
        return GenericResponse(
            status="success",
            response=response
        )
        
    except Exception as e:
        user_language = data.user_language or "english"
        logger.error(f"CNIC verification failure API error: {e}")
        return GenericResponse(
            status="error",
            response=get_localized_response("cnic_not_found", user_language),
            error=str(e)
        )

@router.post("/enhanced_account_selection", response_model=Dict[str, Any])
async def enhanced_account_selection(data: EnhancedAccountSelectionRequest):
    """Enhanced account selection with intelligent understanding and language support."""
    try:
        user_language = data.user_language or detect_user_language(data.user_input)
        
        logger.info({
            "action": "api_enhanced_account_selection",
            "user_input": data.user_input,
            "available_accounts": data.available_accounts,
            "first_name": data.first_name,
            "user_language": user_language
        })
        
        # Use the enhanced account selection method from AI agent
        selection_result = await enhanced_ai_agent.enhanced_account_selection_v2(
            data.user_input, data.available_accounts, data.first_name
        )
        
        # Generate response for account selection guidance if no match
        if not selection_result.get("matched_account"):
            # Use response generator for account selection guidance
            response = await enhanced_ai_agent.response_generator.generate_account_selection_response(
                data.available_accounts, data.user_input, data.first_name, selection_result
            )
            selection_result["response"] = response
        
        selection_result["user_language"] = user_language
        return {
            "status": "success",
            **selection_result
        }
        
    except Exception as e:
        user_language = data.user_language or detect_user_language(data.user_input)
        logger.error(f"Enhanced account selection API error: {e}")
        
        # Generate fallback response
        try:
            fallback_response = await enhanced_ai_agent.response_generator.generate_account_selection_response(
                data.available_accounts, data.user_input, data.first_name, None
            )
        except:
            fallback_response = get_localized_response("account_selection_help", user_language)
        
        return {
            "status": "error",
            "selection_method": "invalid",
            "matched_account": None,
            "response": fallback_response,
            "user_language": user_language,
            "error": str(e)
        }

@router.post("/handle_account_confirmation", response_model=GenericResponse)
async def handle_account_confirmation(data: AccountConfirmationRequest):
    """Handle account confirmation after selection with language support."""
    try:
        user_language = data.user_language or "english"
        
        response = await enhanced_ai_agent.handle_account_confirmation(
            data.selected_account, data.user_name
        )
        
        return GenericResponse(
            status="success",
            response=response
        )
        
    except Exception as e:
        user_language = data.user_language or "english"
        logger.error(f"Account confirmation API error: {e}")
        first_name = data.user_name.split()[0] if data.user_name else ""
        
        if user_language == "roman_urdu":
            fallback = f"Perfect! Account ***-***-{data.selected_account[-4:]} ready hai, {first_name}. Main aapki kaise madad kar sakta hoon?"
        elif user_language == "urdu_script":
            fallback = f"بہترین! اکاؤنٹ ***-***-{data.selected_account[-4:]} تیار ہے، {first_name}۔ میں آپ کی کیسے مدد کر سکتا ہوں؟"
        else:
            fallback = f"Perfect! Account ***-***-{data.selected_account[-4:]} is ready, {first_name}. How can I help you today?"
        
        return GenericResponse(
            status="error",
            response=fallback,
            error=str(e)
        )

@router.post("/handle_session_end", response_model=GenericResponse)
async def handle_session_end(data: SessionEndRequest):
    """Handle session termination with cleanup and language support."""
    try:
        user_language = data.user_language or "english"
        
        response = await enhanced_ai_agent.handle_session_end(
            data.account_number, data.first_name
        )
        
        return GenericResponse(
            status="success",
            response=response
        )
        
    except Exception as e:
        user_language = data.user_language or "english"
        logger.error(f"Session end API error: {e}")
        return GenericResponse(
            status="error",
            response=get_localized_response("session_ended", user_language),
            error=str(e)
        )

@router.post("/generate_error_response", response_model=GenericResponse)
async def generate_error_response(data: ErrorResponseRequest):
    """Generate natural error responses with language support."""
    try:
        user_language = data.user_language or detect_user_language(data.user_message)
        
        # Use the response generator instead of calling enhanced_ai_agent.generate_response directly
        response = await enhanced_ai_agent.response_generator.generate_response(
            "Error occurred", 
            {"error": data.error}, 
            data.user_message, 
            data.first_name, 
            "", 
            "error"
        )
        
        return GenericResponse(
            status="success",
            response=response
        )
        
    except Exception as e:
        user_language = data.user_language or detect_user_language(data.user_message)
        logger.error(f"Error response generation API error: {e}")
        fallback_name = data.first_name if data.first_name else ""
        return GenericResponse(
            status="error",
            response=get_localized_response("technical_error", user_language, name=f", {fallback_name}" if fallback_name else ""),
            error=str(e)
        )

@router.post("/enhanced_query", response_model=EnhancedQueryResponse)
async def enhanced_query(data: EnhancedQueryRequest):
    """NEW: Enhanced query endpoint with detailed processing info and language support."""
    try:
        # Auto-detect language if not provided
        user_language = data.user_language or detect_user_language(data.user_message)
        
        logger.info({
            "action": "api_enhanced_query_start",
            "user_message": data.user_message,
            "account_number": data.account_number,
            "first_name": data.first_name,
            "user_language": user_language
        })
        
        # Get conversation context
        memory = enhanced_ai_agent.get_user_memory(data.account_number)
        conversation_context = enhanced_ai_agent._get_context_summary(memory.chat_memory.messages)
        
        # Classify the query
        query_classification = await enhanced_ai_agent.classify_query_complexity(
            data.user_message, conversation_context, data.account_number
        )
        
        # Process with enhanced hybrid approach
        response = await enhanced_ai_agent.process_transactions_with_context(
            data.user_message, data.account_number, data.first_name
        )
        
        # Determine processing method used
        processing_method = "simple_query"
        if query_classification.requires_pipeline:
            if query_classification.query_type == "contextual":
                processing_method = "contextual_analysis"
            elif query_classification.query_type == "complex":
                processing_method = "pipeline_generation"
            elif query_classification.query_type == "analysis":
                processing_method = "analysis_pipeline"
        
        # Check if context was stored
        context_stored = data.account_number in enhanced_ai_agent.transaction_contexts
        
        logger.info({
            "action": "api_enhanced_query_success",
            "account_number": data.account_number,
            "user_language": user_language,
            "query_type": query_classification.query_type,
            "processing_method": processing_method,
            "context_stored": context_stored
        })
        
        return EnhancedQueryResponse(
            status="success",
            response=response,
            query_classification=query_classification.dict(),
            processing_method=processing_method,
            context_stored=context_stored
        )
        
    except Exception as e:
        user_language = data.user_language or detect_user_language(data.user_message)
        logger.error({
            "action": "api_enhanced_query_error",
            "error": str(e),
            "account_number": data.account_number,
            "user_message": data.user_message,
            "user_language": user_language
        })
        
        return EnhancedQueryResponse(
            status="error",
            response=get_localized_response("technical_error", user_language, name=f", {data.first_name}" if data.first_name else ""),
            error=str(e)
        )

@router.post("/rag_query", response_model=RAGQueryResponse)
async def rag_query(data: RAGQueryRequest):
    """Process RAG queries for bank information with language support."""
    try:
        # Auto-detect language if not provided
        user_language = data.user_language or detect_user_language(data.user_message)
        
        logger.info({
            "action": "api_rag_query_start",
            "user_message": data.user_message,
            "first_name": data.first_name,
            "user_language": user_language
        })
        
        # Use the enhanced AI agent to generate RAG response
        response = await enhanced_ai_agent.generate_rag_response(
            query=data.user_message,
            first_name=data.first_name
        )
        
        # If RAG response is empty or indicates failure, provide fallback
        if not response or "couldn't find relevant information" in response.lower() or "don't have specific information" in response.lower():
            logger.warning("RAG response was empty or indicated failure, providing fallback")
            
            # Generate fallback bank information response
            fallback_response = await enhanced_ai_agent.response_generator.generate_response(
                "Bank information request with fallback",
                {
                    "query": data.user_message,
                    "bank_name": "Best Bank",
                    "services": ["Account Management", "Money Transfers", "Balance Inquiries", "Transaction History"],
                    "availability": "24/7 Digital Banking"
                },
                data.user_message,
                data.first_name,
                "",
                "rag_fallback"
            )
            response = fallback_response
        
        logger.info({
            "action": "api_rag_query_success",
            "first_name": data.first_name,
            "user_language": user_language,
            "response_length": len(response)
        })
        
        return RAGQueryResponse(
            status="success",
            response=response,
            context_used=[]  # Could be enhanced to return actual context chunks used
        )
        
    except Exception as e:
        user_language = data.user_language or detect_user_language(data.user_message)
        logger.error({
            "action": "api_rag_query_error",
            "error": str(e),
            "user_message": data.user_message,
            "user_language": user_language
        })
        
        # Generate a comprehensive fallback response about Best Bank
        try:
            fallback_response = await enhanced_ai_agent.response_generator.generate_response(
                "Bank information request - system fallback",
                {
                    "query": data.user_message,
                    "bank_name": "Best Bank",
                    "error": "RAG system unavailable",
                    "basic_info": "We offer comprehensive banking services including accounts, transfers, and 24/7 support"
                },
                data.user_message,
                data.first_name,
                "",
                "bank_info_fallback"
            )
        except:
            # Ultimate fallback with language support
            if user_language == "roman_urdu":
                fallback_response = f"Best Bank ek comprehensive banking service hai jo account management, money transfers, balance inquiries, aur transaction history provide karta hai, {data.first_name}. Hamare specific services ke baare mein kya jaanna chahte hain?"
            elif user_language == "urdu_script":
                fallback_response = f"بیسٹ بینک ایک جامع بینکنگ سروس ہے جو اکاؤنٹ مینجمنٹ، منی ٹرانسفر، بیلنس انکوائری، اور ٹرانزیکشن ہسٹری فراہم کرتا ہے، {data.first_name}۔ ہماری خدمات کے بارے میں کیا جاننا چاہتے ہیں؟"
            else:
                fallback_response = f"Best Bank is a comprehensive banking service offering account management, money transfers, balance inquiries, and transaction history. We provide 24/7 digital banking services to help you manage your finances securely and conveniently, {data.first_name}. What specific information about our services would you like to know?"
        
        return RAGQueryResponse(
            status="success",  # Return success even with fallback
            response=fallback_response,
            error=str(e)
        )

@router.get("/rag_health")
async def rag_health_check():
    """Check RAG system health."""
    try:
        from rag_system import bank_rag
        
        if not bank_rag:
            return {
                "status": "unhealthy",
                "reason": "RAG system not initialized",
                "documents_loaded": 0,
                "index_ready": False
            }
        
        return {
            "status": "healthy",
            "documents_loaded": len(bank_rag.documents) if bank_rag.documents else 0,
            "index_ready": bank_rag.index is not None,
            "model_loaded": bank_rag.model is not None,
            "document_path": bank_rag.document_path
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "reason": str(e),
            "documents_loaded": 0,
            "index_ready": False
        }

@router.get("/ai_agent_status")
async def ai_agent_status():
    """NEW: Check enhanced AI agent status and capabilities."""
    try:
        return {
            "status": "healthy",
            "agent_type": "EnhancedHybridBankingAIAgent",
            "capabilities": {
                "dual_mode_switching": True,
                "rag_integration": True,
                "enhanced_account_selection": True,
                "query_classification": True,
                "pipeline_generation": True,
                "context_memory": True,
                "transfer_security": True,
                "language_support": True
            },
            "memory_status": {
                "active_users": len(enhanced_ai_agent.user_memories),
                "active_modes": len(enhanced_ai_agent.user_modes),
                "transfer_states": len(enhanced_ai_agent.transfer_states),
                "transaction_contexts": len(enhanced_ai_agent.transaction_contexts)
            },
            "language_features": {
                "supported_languages": ["english", "roman_urdu", "urdu_script"],
                "auto_detection": True,
                "localized_responses": True,
                "fallback_handling": True
            },
            "database_connection": "active",
            "llm_connection": "active"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "agent_type": "EnhancedHybridBankingAIAgent"
        }

# Health check endpoint
@router.get("/health")
async def health_check():
    """Enhanced health check endpoint for monitoring."""
    rag_status = "unknown"
    try:
        from rag_system import bank_rag
        if bank_rag and bank_rag.index and bank_rag.documents:
            rag_status = "healthy"
        else:
            rag_status = "unhealthy"
    except:
        rag_status = "error"
    
    ai_agent_status = "unknown"
    try:
        if enhanced_ai_agent:
            ai_agent_status = "healthy"
        else:
            ai_agent_status = "unhealthy"
    except:
        ai_agent_status = "error"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "pure_api_based_banking_ai_backend",
        "architecture": "api_based_microservices",
        "features": {
            "authentication": "cnic_based_with_account_selection",
            "hybrid_mode": "enhanced_simple_complex_contextual_analysis", 
            "rag_system": rag_status,
            "ai_agent": ai_agent_status,
            "enhanced_account_selection": "multi_format_currency_support",
            "transfer_security": "otp_and_confirmation",
            "thinking_indicator": "2_second_threshold",
            "context_limitation": "strict_banking_only",
            "cnic_text_extraction": "smart_pattern_matching",
            "query_classification": "simple_complex_analysis_contextual",
            "pipeline_generation": "dynamic_mongodb_aggregation",
            "context_memory": "transaction_analysis_storage",
            "api_communication": "pure_http_api_calls_only",
            "service_separation": "clean_microservices_architecture",
            "language_support": "english_roman_urdu_urdu_script_auto_detection"
        },
        "api_endpoints": {
            "authentication": ["verify_cnic", "select_account"],
            "query_processing": ["process_query", "enhanced_query", "rag_query"],
            "ai_operations": ["detect_initial_choice", "enhanced_account_selection", "handle_session_end"],
            "response_generation": ["generate_error_response", "handle_initial_greeting"],
            "utility": ["extract_cnic_from_text", "handle_invalid_cnic_format"],
            "banking": ["user_balance", "transfer_money", "execute_pipeline"],
            "health": ["health", "rag_health", "ai_agent_status"]
        },
        "language_features": {
            "supported_languages": ["english", "roman_urdu", "urdu_script"],
            "auto_detection": "enabled",
            "localized_responses": "enabled",
            "fallback_handling": "enabled"
        },
        "communication_pattern": "webhook_to_api_only",
        "enhanced_hybrid_agent": "backend_only_no_direct_imports",
        "conversation_memory": "persistent_per_session_with_cleanup"
    }