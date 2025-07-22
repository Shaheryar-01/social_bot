import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
import jsonschema
import re
from prompts import (
    filter_extraction_prompt,
    pipeline_generation_prompt,
    response_prompt,
    query_prompt,
    intent_prompt,
    transfer_prompt
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LangChain LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1
)

# MongoDB pipeline schema for validation
PIPELINE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "$match": {"type": "object"},
            "$group": {"type": "object"},
            "$sort": {"type": "object"},
            "$limit": {"type": "integer", "minimum": 1},
            "$project": {"type": "object"}
        },
        "additionalProperties": False
    }
}

class FilterExtraction(BaseModel):
    description: Optional[str] = None
    category: Optional[str] = None
    month: Optional[str] = None
    year: Optional[int] = None
    transaction_type: Optional[str] = None
    amount_range: Optional[Dict[str, float]] = None
    date_range: Optional[Dict[str, str]] = None
    limit: Optional[int] = None
    currency: Optional[str] = None  # Added for new dataset structure

class QueryResult(BaseModel):
    intent: str = Field(default="general")
    pipeline: List[Dict[str, Any]] = Field(default_factory=list)
    response_format: str = Field(default="natural_language")
    filters: Optional[FilterExtraction] = None

class ContextualQuery(BaseModel):
    """Result of contextual query analysis."""
    needs_context: bool = False
    has_reference: bool = False
    is_complete: bool = True
    missing_info: List[str] = Field(default_factory=list)
    clarification_needed: Optional[str] = None
    resolved_query: Optional[str] = None

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
    
    def get_professional_response_prompt(self, user_message: str, intent: str, data: str, first_name: str, 
                                       is_contextual: bool = False) -> str:
        """Generate enhanced professional response prompt."""
        
        time_greeting = self.get_time_of_day_greeting()
        
        # Different prompts for different intents
        if intent == "balance_inquiry":
            return self._get_balance_response_prompt(user_message, data, first_name, time_greeting, is_contextual)
        elif intent in ["spending_analysis", "category_spending"]:
            return self._get_spending_response_prompt(user_message, data, first_name, time_greeting, is_contextual)
        elif intent == "transaction_history":
            return self._get_transaction_response_prompt(user_message, data, first_name, time_greeting, is_contextual)
        elif intent == "transfer_money":
            return self._get_transfer_response_prompt(user_message, data, first_name, time_greeting, is_contextual)
        else:
            return self._get_general_response_prompt(user_message, data, first_name, time_greeting, is_contextual)
    
    def _get_balance_response_prompt(self, user_message: str, data: str, first_name: str, 
                                   time_greeting: str, is_contextual: bool) -> str:
        return f"""
        You are Sage, a warm and professional personal banking assistant. You're helping {first_name}, a valued client.

        User Query: "{user_message}"
        Balance Data: {data}
        
        Response Style:
        - Warm and professional tone
        - Use {first_name}'s name appropriately (not excessively)
        - Start with "{time_greeting}, {first_name}!" if this is a fresh conversation
        - If contextual follow-up, use transitions like "Absolutely!" or "Of course!"
        
        Instructions:
        1. Present the balance information clearly and positively
        2. Handle both PKR and USD accounts appropriately
        3. Add reassuring context about account standing
        4. Offer helpful next steps or additional services
        5. Use formatting for better readability (bullets, sections)
        6. End with a warm offer to help further
        
        Example tone: "I'm pleased to share that your current account balance is excellent. Your account is in great standing, and I'm here if you need any other assistance today."
        
        Make it feel like talking to a trusted personal banker who genuinely cares about helping.
        """
    
    def _get_spending_response_prompt(self, user_message: str, data: str, first_name: str, 
                                    time_greeting: str, is_contextual: bool) -> str:
        return f"""
        You are Sage, a warm and professional personal banking assistant helping {first_name}.

        User Query: "{user_message}"
        Spending Data: {data}
        Is Contextual Follow-up: {is_contextual}
        
        Response Style:
        - Warm, helpful, and insightful
        - If first interaction: "{time_greeting}, {first_name}!"
        - If contextual: "Perfect!" or "Great question!" or "Let me break that down for you!"
        
        Instructions:
        1. Present spending information with helpful context and insights
        2. Handle currency properly (PKR/USD) based on account type
        3. Add percentage calculations where relevant 
        4. Provide spending velocity context (daily/weekly averages)
        5. Compare to typical patterns when possible
        6. Offer proactive follow-up suggestions:
           - "Would you like me to break this down by category?"
           - "I can show you how this compares to last month"
           - "Would you like to see which merchants you spent the most at?"
        7. Use positive, encouraging language even for higher spending
        8. Format clearly with bullets or sections for complex information
        
        Always end with offering additional help or insights.
        """
    
    def _get_transaction_response_prompt(self, user_message: str, data: str, first_name: str, 
                                       time_greeting: str, is_contextual: bool) -> str:
        return f"""
        You are Sage, a professional banking assistant helping {first_name} review their transactions.

        User Query: "{user_message}"
        Transaction Data: {data}
        Is Contextual Follow-up: {is_contextual}
        
        Response Style:
        - Professional but friendly
        - If first interaction: "{time_greeting}, {first_name}!"
        - If contextual: "Here's what I found!" or "Let me pull that up for you!"
        
        Instructions:
        1. Present transactions in a clear, organized format
        2. Handle currency display properly (PKR/USD)
        3. Highlight key insights (largest transaction, most frequent merchant, etc.)
        4. Group related transactions when helpful
        5. Point out any interesting patterns or unusual activity (tactfully)
        6. Offer additional analysis options
        7. Use clear formatting (dates, amounts, descriptions)
        
        Make the transaction review feel like a helpful financial advisor reviewing activity with care and attention.
        """
    
    def _get_transfer_response_prompt(self, user_message: str, data: str, first_name: str, 
                                    time_greeting: str, is_contextual: bool) -> str:
        return f"""
        You are Sage, a trusted banking assistant helping {first_name} with their money transfer.

        User Query: "{user_message}"
        Transfer Result: {data}
        
        Response Style:
        - Professional, reassuring, and detail-oriented
        - Celebratory tone for successful transfers
        - Clear and organized information presentation
        
        Instructions:
        1. Confirm transfer success with enthusiasm ("Excellent!" or "Perfect!")
        2. Present transfer details in organized format including currency
        3. Provide helpful timing information
        4. Reassure about security and completion
        5. Show new balance in correct currency
        6. Offer additional assistance
        
        End with offering further assistance.
        """
    
    def _get_general_response_prompt(self, user_message: str, data: str, first_name: str, 
                                   time_greeting: str, is_contextual: bool) -> str:
        return f"""
        You are Sage, a helpful banking assistant. Respond to {first_name}'s query in a warm, professional manner.

        User Query: "{user_message}"
        Available Information: {data}
        
        Provide a helpful, professional response that offers guidance and additional assistance.
        Use {time_greeting}, {first_name} as appropriate.
        """

def month_to_number(month: str) -> int:
    """Convert month name to number."""
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    return months.get(month.lower(), 1)

def month_days(month: str, year: int) -> int:
    """Get number of days in a month."""
    month_num = month_to_number(month)
    if month_num in [4, 6, 9, 11]:
        return 30
    elif month_num == 2:
        return 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
    else:
        return 31
        
_BRACE_RE = re.compile(r'[{[]')

def _find_json_span(text: str) -> Tuple[int, int]:
    """Return (start, end) indices of the first JSON value in text."""
    m = _BRACE_RE.search(text)
    if not m:
        raise ValueError("No '{' or '[' found")
    start = m.start()
    stack = [text[start]]
    for i in range(start + 1, len(text)):
        ch = text[i]
        if ch in '{[':
            stack.append(ch)
        elif ch in '}]':
            if not stack:
                break
            open_ch = stack.pop()
            if not stack:
                return start, i + 1
    raise ValueError("Unbalanced brackets")

def _json_fix(raw: str) -> str:
    """Best‑effort clean‑ups that keep strict JSON subset."""
    fixed = raw.strip()
    fixed = re.sub(r"'", '"', fixed)
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    fixed = fixed.replace('NaN', 'null')
    fixed = fixed.replace('Infinity', '1e308')
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', fixed)
    return fixed

class BankingAIAgent:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        # Use LangChain memory directly without ConversationChain
        self.user_memories: Dict[str, ConversationBufferMemory] = {}
        self.response_formatter = ProfessionalResponseFormatter()
        
    def get_user_memory(self, account_number: str) -> ConversationBufferMemory:
        """Get or create conversation memory for a user account."""
        if account_number not in self.user_memories:
            self.user_memories[account_number] = ConversationBufferMemory(
                return_messages=True
            )
        return self.user_memories[account_number]
    
    def analyze_contextual_query(self, user_message: str, account_number: str) -> ContextualQuery:
        """Analyze if query needs context using conversation history."""
        memory = self.get_user_memory(account_number)
        
        # Get conversation history
        chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
        
        # First, use LLM to detect if this is a contextual query
        contextual_detection_result = self._detect_contextual_reference_with_llm(user_message, chat_history)
    
        # If no contextual reference detected, analyze as standalone query
        if not contextual_detection_result["is_contextual"]:
            return self._analyze_standalone_query(user_message)
    
        # If contextual reference detected but no previous conversation exists
        if not chat_history:
            return ContextualQuery(
                needs_context=True,
                has_reference=True,
                is_complete=False,
                clarification_needed="I don't have any previous conversation to reference. Could you please provide the complete information for your request?"
            )
    
        # Try to resolve the query with context
        try:
            resolved_query = self._resolve_contextual_query_with_llm(user_message, chat_history)
            return ContextualQuery(
                needs_context=True,
                has_reference=True,
                is_complete=True,
                resolved_query=resolved_query
            )
        except Exception as e:
            logger.error(f"Error resolving contextual query: {e}")
            return ContextualQuery(
                needs_context=True,
                has_reference=True,
                is_complete=False,
                clarification_needed="I couldn't understand the context. Could you please provide the complete information for your request?"
            )

    def _detect_contextual_reference_with_llm(self, user_message: str, chat_history: List) -> Dict[str, Any]:
        """Use LLM to detect if user query references previous context."""
        context_summary = "No previous conversation available"
        if chat_history:
            recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
            context_summary = "\n".join([
                f"{'Human' if i % 2 == 0 else 'Assistant'}: {msg.content}" 
                for i, msg in enumerate(recent_messages)
            ])
    
        contextual_detection_prompt = f"""
        Analyze if the current user query is referencing or building upon previous conversation context.

        Current Query: "{user_message}"
        Previous Conversation: {context_summary}

        A query is contextual if it:
        1. References previous results (e.g., "from this", "from that data", "those transactions")
        2. Uses pronouns that refer to previous content (e.g., "them", "these", "it")
        3. Asks for filtering/drilling down into previous results
        4. Uses relative terms that depend on previous context
        5. Asks follow-up questions that only make sense with previous context

        Return JSON: {{"is_contextual": true/false, "confidence": 0.0-1.0, "reasoning": "explanation"}}
        """
    
        try:
            response = llm.invoke([SystemMessage(content=contextual_detection_prompt)])
            result = self.extract_json_from_response(response.content)
        
            if result and isinstance(result, dict):
                return result
            else:
                return {"is_contextual": False, "confidence": 0.5, "reasoning": "Could not parse LLM response"}
            
        except Exception as e:
            logger.error(f"Error in contextual detection with LLM: {e}")
            return self._fallback_trigger_word_detection(user_message)

    def _fallback_trigger_word_detection(self, user_message: str) -> Dict[str, Any]:
        """Fallback method using trigger words if LLM fails."""
        context_phrases = [
            "from this", "from that", "out of this", "out of that", "from the above",
            "from these", "from those", "of this", "of that", "in this", "in that",
            "them", "these", "those", "it", "they", "break it down", "filter them",
            "show me the", "which ones", "the highest", "the lowest", "the recent ones"
        ]
    
        has_reference = any(phrase in user_message.lower() for phrase in context_phrases)
    
        return {
            "is_contextual": has_reference,
            "confidence": 0.7 if has_reference else 0.8,
            "reasoning": f"Trigger word detection: {'found' if has_reference else 'not found'} contextual phrases"
        }

    def _resolve_contextual_query_with_llm(self, user_message: str, chat_history: List) -> str:
        """Enhanced contextual query resolution using LLM."""
        recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
        conversation_context = "\n".join([
            f"{'Human' if i % 2 == 0 else 'Assistant'}: {msg.content}" 
            for i, msg in enumerate(recent_messages)
        ])
    
        resolution_prompt = f"""
        You are helping resolve a contextual banking query. The user is referencing previous conversation context.
        Current User Query: "{user_message}"
        Previous Conversation: {conversation_context}
        
        Create a complete, standalone query that combines current query with previous context.
        Return ONLY the resolved query as a plain string, no JSON or formatting.
        """
    
        try:
            response = llm.invoke([SystemMessage(content=resolution_prompt)])
            resolved_query = response.content.strip().strip('"\'')
            return resolved_query
        except Exception as e:
            logger.error(f"Error resolving contextual query with LLM: {e}")
            raise e

    def _analyze_standalone_query(self, user_message: str) -> ContextualQuery:
        """Simplified analysis - only check transfers for completeness."""
        if any(word in user_message.lower() for word in ["transfer", "send", "pay", "wire", "remit"]):
            completeness_prompt = f"""
            Analyze this transfer query for completeness:
            Query: "{user_message}"
            
            Check if the query has:
            1. Amount (e.g., 500, 1000 PKR, 50 USD, half of that, etc.)
            2. Recipient (e.g., "to John", "to account 1234", etc.)
            
            Return JSON: {{"is_complete": true/false, "missing_info": ["amount", "recipient"], "clarification_needed": "What to ask for"}}
            """
        
            try:
                response = llm.invoke([SystemMessage(content=completeness_prompt)])
                result = self.extract_json_from_response(response.content)
            
                if result:
                    return ContextualQuery(
                        needs_context=False,
                        has_reference=False,
                        is_complete=result.get("is_complete", True),
                        missing_info=result.get("missing_info", []),
                        clarification_needed=result.get("clarification_needed")
                    )
            except Exception as e:
                logger.error(f"Error analyzing transfer completeness: {e}")
    
        return ContextualQuery(is_complete=True)
    
    def extract_json_from_response(self, raw: str) -> Optional[Any]:
        """Extract the first JSON value from an LLM reply."""
        try:
            start, end = _find_json_span(raw)
            candidate = raw[start:end]
        except ValueError as e:
            logger.error({"action": "extract_json_span_fail", "error": str(e)})
            return None

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        candidate = _json_fix(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.error({"action": "extract_json_parse_fail", "error": str(e), "candidate": candidate[:200]})
            return None
    
    def extract_filters_with_llm(self, user_message: str) -> FilterExtraction:
        """Use LLM to extract filters from user query with new dataset structure."""
        try:
            response = llm.invoke([SystemMessage(content=filter_extraction_prompt.format(
                user_message=user_message,
                current_date=datetime.now().strftime("%Y-%m-%d")
            ))])
            
            try:
                filters_obj = self.extract_json_from_response(response.content)
                if filters_obj is None:
                    raise ValueError("Could not parse filter JSON")
                filters = FilterExtraction(**filters_obj)
                return filters
            
            except (json.JSONDecodeError, TypeError) as e:
                logger.error({
                    "action": "filter_extraction_parse_error",
                    "error": str(e),
                    "raw_response": response.content
                })
                return FilterExtraction()
                
        except Exception as e:
            logger.error({
                "action": "extract_filters_with_llm",
                "error": str(e)
            })
            return FilterExtraction()
        
    def generate_pipeline_from_filters(self, filters: FilterExtraction, intent: str, account_number: str) -> List[Dict[str, Any]]:
        """Generate MongoDB pipeline from extracted filters using LLM."""
        try:
            response = llm.invoke([
                SystemMessage(content=pipeline_generation_prompt.format(
                    filters=json.dumps(filters.dict()),
                    intent=intent,
                    account_number=account_number
                ))
            ])
            
            cleaned_response = self.extract_json_from_response(response.content)
        
            if not cleaned_response:
                return self._generate_fallback_pipeline(filters, intent, account_number)
            
            pipeline = cleaned_response
            jsonschema.validate(pipeline, PIPELINE_SCHEMA)
            return pipeline
            
        except Exception as e:
            logger.error({
                "action": "generate_pipeline_from_filters",
                "error": str(e)
            })
            return self._generate_fallback_pipeline(filters, intent, account_number)
    
    def _generate_fallback_pipeline(self, filters: FilterExtraction, intent: str, account_number: str) -> List[Dict[str, Any]]:
        """Generate a basic pipeline when LLM fails."""
        match_stage = {"$match": {"account_number": account_number}}
        
        if intent == "transaction_history":
            pipeline = [match_stage, {"$sort": {"date": -1, "_id": -1}}]
            if filters.limit:
                pipeline.append({"$limit": filters.limit})
            return pipeline
        
        elif intent in ["spending_analysis", "category_spending"]:
            if filters.transaction_type:
                match_stage["$match"]["type"] = filters.transaction_type
            
            if filters.description:
                match_stage["$match"]["description"] = {
                    "$regex": filters.description,
                    "$options": "i"
                }
            
            if filters.category:
                match_stage["$match"]["category"] = {
                    "$regex": filters.category,
                    "$options": "i"
                }
            
            pipeline = [
                match_stage,
                {
                    "$group": {
                        "_id": None,
                        "total_amount": {"$sum": "$transaction_amount"},
                        "currency": {"$first": "$transaction_currency"}
                    }
                }
            ]
            return pipeline
        
        return [match_stage, {"$sort": {"date": -1, "_id": -1}}, {"$limit": 10}]

    def detect_intent_from_filters(self, user_message: str, filters: FilterExtraction) -> str:
        """Detect intent using LLM for more flexible understanding."""
        try:
            response = llm.invoke([
                SystemMessage(content=intent_prompt.format(
                    user_message=user_message,
                    filters=json.dumps(filters.dict())
                ))
            ])
            
            detected_intent = response.content.strip().lower()
            
            valid_intents = [
                "balance_inquiry",
                "transaction_history", 
                "spending_analysis",
                "category_spending",
                "transfer_money",
                "general"
            ]
            
            if detected_intent in valid_intents:
                return detected_intent
            else:
                for intent in valid_intents:
                    if intent in detected_intent:
                        return intent
                return "general"
                
        except Exception as e:
            logger.error({
                "action": "llm_intent_classification",
                "error": str(e),
                "user_message": user_message
            })
            return self._rule_based_intent_fallback(user_message, filters)

    def _rule_based_intent_fallback(self, user_message: str, filters: FilterExtraction) -> str:
        """Enhanced rule-based intent detection."""
        user_message_lower = user_message.lower()
        
        # Enhanced keywords for better detection
        balance_keywords = ["balance", "money", "amount", "funds", "account", "cash", "afford", "target", "save", "purchase", "buy", "enough", "capacity"]
        transaction_keywords = ["transaction", "history", "recent", "last", "show", "list", "activities"]
        spending_keywords = ["spend", "spent", "spending", "expenditure", "expense", "expenses", 
                            "cost", "costs", "paid", "pay", "payment", "purchase", "purchased", 
                            "buying", "bought", "money went", "charged", "compare", "more than", 
                            "less than", "patterns", "habits", "analysis", "right now"]
        transfer_keywords = ["transfer", "send", "wire", "remit", "move money", "i want to transfer", 
                            "transfer money", "send money", "pay"]
        planning_keywords = ["planning", "target", "goal", "save", "afford", "can i", "what can i do"]
        
        # Check for transfer intent first (highest priority for explicit transfer requests)
        if any(keyword in user_message_lower for keyword in transfer_keywords):
            return "transfer_money"
        
        # Financial planning and affordability questions
        if any(keyword in user_message_lower for keyword in planning_keywords):
            return "balance_inquiry"
        
        # Spending comparisons and analysis
        if any(keyword in user_message_lower for keyword in ["more than", "less than", "compared to", "compare", "vs", "versus"]):
            return "spending_analysis"
            
        # Traditional keyword matching
        if any(keyword in user_message_lower for keyword in balance_keywords):
            return "balance_inquiry"
        elif any(keyword in user_message_lower for keyword in transaction_keywords) or filters.limit:
            return "transaction_history"
        elif any(keyword in user_message_lower for keyword in spending_keywords):
            if filters.category:
                return "category_spending"
            else:
                return "spending_analysis"
        else:
            return "general"

    def detect_intent_fallback(self, user_message: str) -> tuple[str, List[Dict[str, Any]]]:
        """Improved fallback intent detection using LLM filter extraction."""
        filters = self.extract_filters_with_llm(user_message)
        intent = self.detect_intent_from_filters(user_message, filters)
        pipeline = self.generate_pipeline_from_filters(filters, intent, "{{account_number}}")
        
        return intent, pipeline

    def replace_account_number_in_pipeline(self, pipeline: List[Dict[str, Any]], account_number: str) -> List[Dict[str, Any]]:
        """Recursively replace {{account_number}} placeholder in pipeline."""
        def replace_in_dict(obj):
            if isinstance(obj, dict):
                return {k: replace_in_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_in_dict(item) for item in obj]
            elif isinstance(obj, str):
                return obj.replace("{{account_number}}", account_number)
            else:
                return obj
        
        return replace_in_dict(pipeline)

    async def process_query(self, user_message: str, account_number: str, first_name: str) -> str:
        """Process user banking queries with LangChain ConversationBufferMemory."""
        memory = self.get_user_memory(account_number)

        # Step 1: Analyze if query needs context or clarification
        contextual_analysis = self.analyze_contextual_query(user_message, account_number)
        
        if not contextual_analysis.is_complete and contextual_analysis.clarification_needed:
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(contextual_analysis.clarification_needed)
            return contextual_analysis.clarification_needed
        
        # Step 2: Use resolved query if available, otherwise use original
        query_to_process = contextual_analysis.resolved_query or user_message

        # Step 3: Analyze the query (resolved or original)
        query_analysis = await self._analyze_query(query_to_process, account_number)

        # Step 4: Validate pipeline
        if query_analysis.pipeline:
            try:
                jsonschema.validate(query_analysis.pipeline, PIPELINE_SCHEMA)
            except jsonschema.ValidationError as e:
                error_message = "Error: Invalid MongoDB pipeline generated. Please try rephrasing your query."
                memory.chat_memory.add_user_message(user_message)
                memory.chat_memory.add_ai_message(error_message)
                return error_message

        # Step 5: Execute appropriate action
        response = None
        
        if query_analysis.intent == "balance_inquiry":
            response = await self._handle_balance_inquiry(account_number, first_name, query_analysis, query_to_process, memory)
        elif query_analysis.intent in ["transaction_history", "spending_analysis", "category_spending"]:
            response = await self._handle_data_query(account_number, query_analysis, query_to_process, first_name, memory)
        elif query_analysis.intent == "transfer_money":
            response = await self._handle_money_transfer(account_number, query_analysis, query_to_process, first_name, memory)
        else:
            response = await self._handle_general_query(query_to_process, first_name, memory)

        # Step 6: Add conversation to memory
        if response:
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)

        return response

    async def _analyze_query(self, user_message: str, account_number: str) -> QueryResult:
        """Use LLM to analyze query and generate MongoDB pipeline."""
        try:
            intent, pipeline = self.detect_intent_fallback(user_message)
            if intent != "general":
                pipeline = self.replace_account_number_in_pipeline(pipeline, account_number)
                return QueryResult(intent=intent, pipeline=pipeline)
        except Exception as e:
            logger.error({
                "action": "fallback_intent_detection",
                "error": str(e),
                "user_message": user_message
            })

        try:
            response = llm.invoke([
                SystemMessage(content=query_prompt.format(
                    user_message=user_message,
                    current_date=datetime.now().strftime("%Y-%m-%d")
                ))
            ])

            result = self.extract_json_from_response(response.content)
            if result is None:
                return QueryResult(intent="general", pipeline=[])

            if not isinstance(result, dict) or "intent" not in result:
                return QueryResult(intent="general", pipeline=[])

            pipeline = self.replace_account_number_in_pipeline(result.get("pipeline", []), account_number)

            query_result = QueryResult(
                intent=result.get("intent", "general"),
                pipeline=pipeline,
                response_format=result.get("response_format", "natural_language")
            )
            return query_result
        except Exception as e:
            logger.error({
                "action": "analyze_query",
                "error": str(e),
                "user_message": user_message
            })
            return QueryResult(intent="general", pipeline=[])

    async def _handle_balance_inquiry(self, account_number: str, first_name: str, query_analysis: QueryResult, 
                                     user_message: str, memory: ConversationBufferMemory) -> str:
        """Enhanced balance inquiry with professional responses and memory integration."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/user_balance",
                    json={"account_number": account_number}
                )
                response.raise_for_status()
                data = response.json()
                
                chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
                is_contextual = len(chat_history) > 0
                
                professional_prompt = self.response_formatter.get_professional_response_prompt(
                    user_message=user_message,
                    intent="balance_inquiry",
                    data=json.dumps(data),
                    first_name=first_name,
                    is_contextual=is_contextual
                )
                
                formatted_response = llm.invoke([SystemMessage(content=professional_prompt)])
                return formatted_response.content
                
        except Exception as e:
            logger.error({"action": "handle_balance_inquiry", "error": str(e)})
            return f"I apologize, {first_name}, but I'm experiencing a technical issue retrieving your balance. Please try again in a moment, and I'll be happy to help!"

    async def _handle_data_query(self, account_number: str, query_analysis: QueryResult, user_message: str, 
                                first_name: str, memory: ConversationBufferMemory) -> str:
        """Enhanced data query handling with professional responses and memory integration."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/execute_pipeline",
                    json={"account_number": account_number, "pipeline": query_analysis.pipeline}
                )
                response.raise_for_status()
                data = response.json()
                
                chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
                is_contextual = len(chat_history) > 0
                
                professional_prompt = self.response_formatter.get_professional_response_prompt(
                    user_message=user_message,
                    intent=query_analysis.intent,
                    data=json.dumps(data),
                    first_name=first_name,
                    is_contextual=is_contextual
                )
                
                formatted_response = llm.invoke([SystemMessage(content=professional_prompt)])
                return formatted_response.content
                
        except Exception as e:
            logger.error({"action": "handle_data_query", "error": str(e)})
            return "I apologize, but I encountered an issue processing your request. Please try again, and I'll be happy to help you with your query!"

    async def _handle_money_transfer(self, account_number: str, query_analysis: QueryResult, user_message: str, 
                                   first_name: str, memory: ConversationBufferMemory) -> str:
        """Enhanced money transfer with better incomplete transfer handling."""
        try:
            # Enhanced transfer parsing with contextual amount support
            transfer_prompt_enhanced = f"""
            Extract transfer details from this query, handling contextual references:
            Query: "{user_message}"
            
            Context: Previous conversation may have mentioned amounts. Handle phrases like:
            - "I want to transfer 1000 PKR" (amount specified, recipient missing)
            - "transfer half of that" (where "that" refers to a previous amount)
            - "send 50% of PKR 134,761.10"
            - "transfer $20 to John"
            
            Extract:
            - amount: number (calculate if percentage/fraction given)
            - currency: "PKR" or "USD" (default PKR if not specified)
            - recipient: string (null if not mentioned)
            - has_amount: boolean (true if amount specified)
            - has_recipient: boolean (true if recipient specified)
            
            Return JSON: {{"amount": number, "currency": string, "recipient": string, "has_amount": boolean, "has_recipient": boolean}}
            
            Examples:
            "I want to transfer 1000 PKR" → {{"amount": 1000, "currency": "PKR", "recipient": null, "has_amount": true, "has_recipient": false}}
            "transfer $20 to John" → {{"amount": 20, "currency": "USD", "recipient": "John", "has_amount": true, "has_recipient": true}}
            "send half of that to Alice" → {{"amount": null, "currency": "PKR", "recipient": "Alice", "has_amount": false, "has_recipient": true}}
            "transfer money" → {{"amount": null, "currency": "PKR", "recipient": null, "has_amount": false, "has_recipient": false}}
            """
            
            response = llm.invoke([SystemMessage(content=transfer_prompt_enhanced)])
            transfer_details = self.extract_json_from_response(response.content)
            
            if transfer_details is None:
                return f"I'm sorry, {first_name}, but I couldn't understand the transfer details. Could you please specify the amount, currency (USD or PKR), and recipient? For example: 'Transfer 500 PKR to John Smith'."

            # Check what information is missing and ask specifically
            missing_parts = []
            
            if not transfer_details.get("has_amount") or not transfer_details.get("amount"):
                missing_parts.append("amount")
            
            if not transfer_details.get("has_recipient") or not transfer_details.get("recipient"):
                missing_parts.append("recipient")
            
            # Handle missing information
            if missing_parts:
                if "recipient" in missing_parts and "amount" not in missing_parts:
                    # Amount provided, recipient missing
                    amount = transfer_details.get("amount", 0)
                    currency = transfer_details.get("currency", "PKR")
                    return f"I understand you want to transfer {amount} {currency}. Could you please tell me who you'd like to send it to? For example: 'to John Smith' or just 'John'."
                
                elif "amount" in missing_parts and "recipient" not in missing_parts:
                    # Recipient provided, amount missing
                    recipient = transfer_details.get("recipient", "")
                    return f"I see you want to transfer money to {recipient}. Could you please specify the amount? For example: '500 PKR' or '$20'."
                
                else:
                    # Both missing
                    return f"To process your transfer, {first_name}, I need both the amount and recipient. For example: 'Transfer 500 PKR to John Smith'."
            
            # All information available - process the transfer
            amount = transfer_details.get("amount")
            currency = transfer_details.get("currency", "PKR")
            recipient = transfer_details.get("recipient")
            
            if amount <= 0:
                return f"The transfer amount must be greater than 0. Please specify a valid amount to transfer to {recipient}."

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/transfer_money",
                    json={
                        "from_account": account_number,
                        "to_recipient": recipient,
                        "amount": amount,
                        "currency": currency
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                professional_prompt = self.response_formatter.get_professional_response_prompt(
                    user_message=f"Transfer {amount} {currency} to {recipient}",
                    intent="transfer_money",
                    data=json.dumps(data),
                    first_name=first_name,
                    is_contextual=False
                )
                
                formatted_response = llm.invoke([SystemMessage(content=professional_prompt)])
                return formatted_response.content
                
        except Exception as e:
            logger.error({"action": "handle_money_transfer", "error": str(e)})
            return f"I apologize, {first_name}, but I encountered an issue processing your transfer. Your account security is our priority, so please try again or contact support if the issue persists."

    async def _handle_general_query(self, user_message: str, first_name: str, memory: ConversationBufferMemory) -> str:
        """Enhanced general query handling with professional tone and memory integration."""
        time_greeting = self.response_formatter.get_time_of_day_greeting()
        
        chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
        has_history = len(chat_history) > 0
        
        if has_history:
            greeting = f"Of course, {first_name}!"
        else:
            greeting = f"{time_greeting}, {first_name}!"
        
        help_message = f"""{greeting} I'd be happy to help you with your banking needs.
        
I can assist you with:

💰 **Account Information**
• Check your current balance
• Review recent account activity

📊 **Spending Analysis** 
• Analyze your spending patterns
• Break down expenses by category
• Compare spending across different time periods

📝 **Transaction History**
• View your recent transactions
• Filter transactions by date or amount
• Search for specific merchants or categories

💸 **Money Transfers**
• Transfer funds to other accounts
• Send money to friends and family

You can ask me questions like:
• "What's my current balance?"
• "How much did I spend on groceries last month?"
• "Show me my transactions from this week"
• "Transfer 500 PKR to John Smith"

I'm also great with follow-up questions! After I show you information, you can ask "from this, how much on utilities?" or similar contextual questions.

What would you like to know about your account today?
"""
        
        return help_message