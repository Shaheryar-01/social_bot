import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
import os
from language_utils import detect_user_language, get_language_instruction, get_language_aware_prompt_prefix

logger = logging.getLogger(__name__)

class LLMResponseGenerator:
    """Enhanced LLM Response Generator with Multi-Language Support for Hybrid Banking AI."""
    
    def __init__(self, llm_client=None):
        if llm_client is None:
            # Initialize with ChatOpenAI if no client provided
            self.llm = ChatOpenAI(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3
            )
        else:
            self.llm = llm_client
    
    async def generate_response(self, context: str, data: Any, user_message: str, first_name: str, 
                              conversation_history: str = "", response_type: str = "general") -> str:
        """
        Enhanced response generation with multi-language support for hybrid banking AI.
        
        Args:
            context: What happened (e.g., "Simple query completed", "Complex pipeline generated")
            data: The actual data/results (balance, transactions, analysis results, etc.)
            user_message: Original user message
            first_name: User's first name
            conversation_history: Previous conversation context
            response_type: Type of response (simple_transactions, complex_analysis, contextual_analysis, etc.)
        """
        
        # Language detection and instruction
        language_prefix = get_language_aware_prompt_prefix(user_message)
        
        # Enhanced system prompt for hybrid approach with language support
        system_prompt = f"""{language_prefix}

You are Sage, Best Bank's intelligent and professional banking assistant with enhanced hybrid processing capabilities.

CONTEXT: {context}
USER'S NAME: {first_name}
USER'S MESSAGE: "{user_message}"
RESPONSE TYPE: {response_type}
CONVERSATION HISTORY: {conversation_history}

DATA PROVIDED: {json.dumps(data, default=str) if data else "No specific data"}

CRITICAL INSTRUCTIONS FOR ENHANCED HYBRID SYSTEM:
1. Generate NATURAL, CONVERSATIONAL responses - never robotic or templated
2. Use the provided data accurately but present it naturally
3. Be professional yet warm and helpful
4. For transaction lists, use NUMBERS (1. 2. 3.) instead of bullets (•) - this is CRITICAL
5. For amounts, include currency clearly (PKR/USD)
6. Vary your language - don't repeat the same phrases
7. Match the user's tone - formal or casual as appropriate
8. Keep responses CONCISE and under 1800 characters for messaging platforms
9. Always be accurate with the provided data
10. Make it feel like a human conversation, not a bot response

ENHANCED RESPONSE GUIDELINES BY TYPE:
- simple_transactions: "Here are your transactions:" then numbered list (1. 2. 3.)
- complex_transactions: "Based on your criteria, I found:" then numbered results
- complex_analysis: "Analysis complete! Here's what I found:" then key insights
- contextual_analysis: "Looking at the previous results:" then specific analysis
- balance: Present balance naturally, consider affordability questions
- rag: Answer using bank documentation context, cite naturally
- transfer: Guide through secure process conversationally
- error: Be apologetic but helpful, suggest next steps
- greeting: Be welcoming, present options naturally
- decline: Politely decline non-banking queries, redirect
- account_confirmation: Confirm account selection warmly
- session_end: Provide secure farewell

FORMATTING RULES:
- Transaction lists: Use numbers (1. Date | Description | Type Amount Currency)
- Analysis results: Present insights clearly and concisely
- Context references: "From your previous query..." or "Looking at those transactions..."
- Currency: Always specify PKR or USD clearly
- Dates: Format as "Jan 15, 2025" for readability
- Amounts: Use commas for thousands (1,500 PKR)

HYBRID PROCESSING AWARENESS:
- Simple queries: "I quickly found..." or "Here are..."
- Complex queries: "I analyzed your request..." or "Based on your criteria..."
- Contextual queries: "From your previous results..." or "Looking at those transactions..."

Generate a natural, helpful response based on the context and data provided:"""

        try:
            response = await self.llm.ainvoke([SystemMessage(content=system_prompt)])
            generated_response = response.content.strip()
            
            # Log for debugging
            user_language = detect_user_language(user_message)
            logger.info(f"Enhanced LLM Response Generated - Type: {response_type}, Language: {user_language}, Length: {len(generated_response)}")
            
            return generated_response
            
        except Exception as e:
            logger.error(f"Error generating enhanced LLM response: {e}")
            # Even error responses should be generated by LLM, but as fallback:
            return await self._generate_error_response(first_name, str(e), user_message)
    
    async def _generate_error_response(self, first_name: str, error_details: str, user_message: str = "") -> str:
        """Generate error responses via LLM with language support."""
        
        language_prefix = get_language_aware_prompt_prefix(user_message) if user_message else ""
        
        error_prompt = f"""{language_prefix}

Generate a natural, apologetic response for a technical error in the banking system.
        
USER'S NAME: {first_name}
ERROR: {error_details}
        
Be apologetic, professional, and suggest they try again or contact support.
Keep it natural and conversational. Don't mention technical details."""
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=error_prompt)])
            return response.content.strip()
        except:
            # Absolute fallback with basic language detection
            user_language = detect_user_language(user_message) if user_message else "english"
            if user_language == "roman_urdu":
                return f"Technical problem aa rahi hai{' ' + first_name if first_name else ''}. Thoda baad try kariye please."
            elif user_language == "urdu_script":
                return f"تکنیکی مسئلہ ہو رہا ہے{' ' + first_name if first_name else ''}۔ براہ کرم تھوڑی دیر بعد کوشش کریں۔"
            else:
                return f"I'm experiencing technical difficulties{', ' + first_name if first_name else ''}. Please try again in a moment."
    
    async def generate_rag_response(self, rag_context: str, user_query: str, first_name: str, 
                                  relevant_chunks: List[Dict] = None) -> str:
        """Generate RAG-based responses using LLM with enhanced context handling and language support."""
        
        language_prefix = get_language_aware_prompt_prefix(user_query)
        
        rag_prompt = f"""{language_prefix}

You are answering a banking question using Best Bank's official documentation.

USER'S QUESTION: "{user_query}"
USER'S NAME: {first_name}

CONTEXT FROM BEST BANK DOCUMENTATION:
{rag_context}

RELEVANT CHUNKS: {json.dumps(relevant_chunks, default=str) if relevant_chunks else "No chunks"}

ENHANCED RAG INSTRUCTIONS:
1. Answer the user's question using ONLY the provided context from Best Bank's documentation
2. Be natural and conversational - don't sound like you're reading from a manual
3. If the context doesn't fully answer their question, say so naturally
4. Be professional but warm
5. Make the information easy to understand
6. If there are specific numbers, rates, or procedures, mention them clearly
7. Reference the documentation naturally ("According to our policies..." or "Our bank offers...")
8. If multiple related topics are in context, organize the response logically

Generate a natural, helpful response based on the documentation context:"""

        try:
            response = await self.llm.ainvoke([SystemMessage(content=rag_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            user_language = detect_user_language(user_query)
            if user_language == "roman_urdu":
                return f"Documentation access mein problem aa rahi hai{', ' + first_name if first_name else ''}. Please dobara try kariye ya account access ke liye kahiye."
            elif user_language == "urdu_script":
                return f"دستاویزات تک رسائی میں مسئلہ ہو رہا ہے{' ' + first_name if first_name else ''}۔ براہ کرم دوبارہ کوشش کریں یا اکاؤنٹ تک رسائی کے لیے کہیں۔"
            else:
                return f"I'm having trouble accessing our documentation right now{', ' + first_name if first_name else ''}. Please try again or let me help you with your account instead."
    
    async def generate_transaction_response(self, transactions: List[Dict], user_query: str, 
                                          first_name: str, query_type: str = "simple", 
                                          analysis_result: Dict = None) -> str:
        """Generate enhanced transaction responses for hybrid approach with language support."""
        
        language_prefix = get_language_aware_prompt_prefix(user_query)
        
        if not transactions:
            no_transactions_prompt = f"""{language_prefix}

Generate a natural response when no transactions are found.
            
USER'S NAME: {first_name}
USER'S QUERY: "{user_query}"
QUERY TYPE: {query_type}
            
Be natural and helpful, suggest alternatives or checking different criteria."""
            
            try:
                response = await self.llm.ainvoke([SystemMessage(content=no_transactions_prompt)])
                return response.content.strip()
            except:
                user_language = detect_user_language(user_query)
                if user_language == "roman_urdu":
                    return f"Koi transactions nahi mile aapke criteria ke hisaab se{', ' + first_name if first_name else ''}. Kya alag search terms try karna chahenge?"
                elif user_language == "urdu_script":
                    return f"آپ کے معیار کے مطابق کوئی لین دین نہیں ملا{' ' + first_name if first_name else ''}۔ کیا مختلف تلاش کی شرائط آزمانا چاہیں گے؟"
                else:
                    return f"I don't see any transactions matching your criteria{', ' + first_name if first_name else ''}. Would you like to try different search terms?"
        
        # Format transaction data for LLM
        formatted_transactions = []
        for i, tx in enumerate(transactions, 1):
            date_obj = tx.get("date")
            if isinstance(date_obj, datetime):
                date_str = date_obj.strftime("%b %d, %Y")
            else:
                date_str = str(date_obj) if date_obj else "Unknown"
            
            formatted_tx = {
                "number": i,
                "date": date_str,
                "description": tx.get("description", ""),
                "type": tx.get("type", "").title(), 
                "amount": tx.get("transaction_amount", 0),
                "currency": tx.get("transaction_currency", "PKR").upper()
            }
            formatted_transactions.append(formatted_tx)
        
        transaction_prompt = f"""{language_prefix}

Generate a natural response showing transaction results for hybrid banking system.
        
USER'S NAME: {first_name}
USER'S QUERY: "{user_query}"
QUERY TYPE: {query_type}
TRANSACTION COUNT: {len(transactions)}
ANALYSIS RESULT: {json.dumps(analysis_result, default=str) if analysis_result else "No analysis"}
        
TRANSACTIONS: {json.dumps(formatted_transactions)}
        
ENHANCED HYBRID INSTRUCTIONS:
1. Start with natural introduction based on query type:
   - Simple: "Here are your transactions:"
   - Complex: "Based on your criteria, I found:"
   - Contextual: "From your previous results:"
2. Format transactions with NUMBERS (1. 2. 3.) NOT bullets
3. Format: number. Date | Description | Type Amount Currency
4. Make it easy to scan and read - BE CONCISE
5. Be conversational, not robotic
6. If analysis_result provided, highlight key insights
7. If it's a contextual query, reference the previous context
8. IMPORTANT: Keep responses under 1800 characters for Facebook Messenger
9. For long lists, summarize first few and mention total count
10. If more than 8 transactions, show first 5-6 and say "Total: X transactions in [period]"
        
Generate the response:"""
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=transaction_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating transaction response: {e}")
            user_language = detect_user_language(user_query)
            if user_language == "roman_urdu":
                return f"Aapke transactions mil gaye hain lekin display karne mein problem aa rahi hai{', ' + first_name if first_name else ''}. Please dobara try kariye."
            elif user_language == "urdu_script":
                return f"آپ کے لین دین مل گئے ہیں لیکن دکھانے میں مسئلہ ہو رہا ہے{' ' + first_name if first_name else ''}۔ براہ کرم دوبارہ کوشش کریں۔"
            else:
                return f"I found your transactions but had trouble formatting them{', ' + first_name if first_name else ''}. Please try again."
    
    async def generate_analysis_response(self, analysis_result: Dict, user_query: str, 
                                       first_name: str, analysis_type: str, 
                                       context_used: bool = False) -> str:
        """Generate natural analysis responses for hybrid approach with language support."""
        
        language_prefix = get_language_aware_prompt_prefix(user_query)
        
        analysis_prompt = f"""{language_prefix}

Generate a natural response for banking analysis results.
        
USER'S NAME: {first_name}
USER'S QUERY: "{user_query}"
ANALYSIS TYPE: {analysis_type}
CONTEXT USED: {context_used}
ANALYSIS RESULT: {json.dumps(analysis_result, default=str)}
        
ENHANCED ANALYSIS INSTRUCTIONS:
1. Present the analysis naturally and conversationally
2. Highlight the key finding clearly
3. Include relevant details (amount, currency, description, date)
4. If context_used is True, reference previous results
5. Be helpful and informative
6. For max/min: Show the specific transaction details
7. For avg/sum: Show the calculated result clearly
8. Format amounts with proper currency
        
ANALYSIS TYPES:
- max: "Your highest transaction was..."
- min: "Your lowest transaction was..."
- avg: "Your average transaction amount is..."
- sum: "Your total spending was..."
        
Generate the response:"""
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=analysis_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating analysis response: {e}")
            user_language = detect_user_language(user_query)
            if user_language == "roman_urdu":
                return f"Analysis complete ho gaya hai lekin present karne mein problem aa rahi hai{', ' + first_name if first_name else ''}. Please dobara try kariye."
            elif user_language == "urdu_script":
                return f"تجزیہ مکمل ہو گیا ہے لیکن پیش کرنے میں مسئلہ ہو رہا ہے{' ' + first_name if first_name else ''}۔ براہ کرم دوبارہ کوشش کریں۔"
            else:
                return f"I completed the analysis but had trouble presenting it{', ' + first_name if first_name else ''}. Please try again."
    
    async def generate_balance_response(self, balance_data: Dict, user_query: str, first_name: str) -> str:
        """Generate natural balance responses with affordability context and language support."""
        
        language_prefix = get_language_aware_prompt_prefix(user_query)
        
        balance_prompt = f"""{language_prefix}

Generate a natural response for a balance inquiry with enhanced context.
        
USER'S NAME: {first_name}
USER'S QUERY: "{user_query}"
BALANCE DATA: {json.dumps(balance_data)}
        
ENHANCED BALANCE INSTRUCTIONS:
1. Present the balance naturally and conversationally
2. If they asked about affording something, address that specifically
3. Include currency clearly (PKR/USD)
4. Be helpful and positive
5. Format amounts with proper commas for readability
6. If multiple currencies, show both clearly
7. Consider the context of their question (spending, saving, planning)
        
Generate the response:"""
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=balance_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating balance response: {e}")
            user_language = detect_user_language(user_query)
            if user_language == "roman_urdu":
                return f"Aapka balance dekh sakta hoon lekin show karne mein problem aa rahi hai{', ' + first_name if first_name else ''}. Please dobara try kariye."
            elif user_language == "urdu_script":
                return f"آپ کا بیلنس دیکھ سکتا ہوں لیکن دکھانے میں مسئلہ ہو رہا ہے{' ' + first_name if first_name else ''}۔ براہ کرم دوبارہ کوشش کریں۔"
            else:
                return f"I can see your balance but had trouble presenting it{', ' + first_name if first_name else ''}. Please try again."
    
    async def generate_transfer_response(self, transfer_state: Dict, stage: str, user_message: str, 
                                       first_name: str, result: Dict = None) -> str:
        """Generate natural transfer process responses with enhanced security flow and language support."""
        
        language_prefix = get_language_aware_prompt_prefix(user_message)
        
        transfer_prompt = f"""{language_prefix}

Generate a natural response for money transfer process with enhanced security.
        
USER'S NAME: {first_name}
USER'S MESSAGE: "{user_message}"
TRANSFER STAGE: {stage}
TRANSFER STATE: {json.dumps(transfer_state)}
RESULT: {json.dumps(result) if result else "No result"}
        
TRANSFER STAGES:
- info_collection: Getting transfer details (amount, recipient)
- otp_verification: Asking for security code (1-5 digits)
- confirmation: Final confirmation before processing
- completed: Transfer successful
- cancelled: Transfer cancelled by user
        
ENHANCED TRANSFER INSTRUCTIONS:
1. Be natural and conversational throughout the process
2. Guide them through each stage smoothly
3. Be clear about what you need from them
4. For security stages, emphasize safety and professionalism
5. Celebrate successful transfers, handle errors gracefully
6. For info collection, ask for missing pieces naturally
7. For OTP, explain it's for security (1-5 digits)
8. For confirmation, summarize transfer details clearly
        
Generate appropriate response for this stage:"""
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=transfer_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating transfer response: {e}")
            user_language = detect_user_language(user_message)
            if user_language == "roman_urdu":
                return f"Transfer process mein problem aa rahi hai{', ' + first_name if first_name else ''}. Chaliye start over karte hain."
            elif user_language == "urdu_script":
                return f"ٹرانسفر کے عمل میں مسئلہ ہو رہا ہے{' ' + first_name if first_name else ''}۔ آئیے دوبارہ شروع کرتے ہیں۔"
            else:
                return f"I'm having trouble with the transfer process{', ' + first_name if first_name else ''}. Let me help you start over."
    
    async def generate_account_selection_response(self, accounts: List[str], user_input: str, 
                                                first_name: str, selection_result: Dict = None) -> str:
        """Generate natural account selection responses with enhanced multi-format support and language support."""
        
        language_prefix = get_language_aware_prompt_prefix(user_input)
        
        selection_prompt = f"""{language_prefix}

Generate a natural response for account selection with enhanced capabilities.
        
USER'S NAME: {first_name}
USER'S INPUT: "{user_input}"
AVAILABLE ACCOUNTS: {accounts}
SELECTION RESULT: {json.dumps(selection_result) if selection_result else "No result"}
        
ENHANCED SELECTION INSTRUCTIONS:
1. If showing accounts, format with numbers and mask (1. ***-***-1234)
2. Be helpful in explaining selection options (currency, position, digits)
3. If selection successful, confirm and welcome them warmly
4. If selection failed, guide them gently to correct options
5. Be conversational and supportive
6. Mention multiple ways to select: currency (USD/PKR), position (first/second), or last digits
7. Support multiple languages naturally (English/Urdu terms)
        
SELECTION METHODS:
- Currency: "USD account", "PKR account", "dollar account", "rupee account"
- Position: "first account", "second account", "pehla", "doosra"  
- Digits: Last 4 digits like "1234"
        
Generate the response:"""
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=selection_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating account selection response: {e}")
            user_language = detect_user_language(user_input)
            if user_language == "roman_urdu":
                return f"Aapke accounts dikh rahe hain lekin selection mein problem aa rahi hai{', ' + first_name if first_name else ''}. Please last 4 digits use kariye ya 'USD account' ya 'PKR account' kahiye."
            elif user_language == "urdu_script":
                return f"آپ کے اکاؤنٹس دکھ رہے ہیں لیکن انتخاب میں مسئلہ ہو رہا ہے{' ' + first_name if first_name else ''}۔ براہ کرم آخری 4 ہندسے استعمال کریں یا 'USD account' یا 'PKR account' کہیں۔"
            else:
                return f"I can see your accounts but had trouble with the selection{', ' + first_name if first_name else ''}. Please try using the last 4 digits of your account number or specify 'USD account' or 'PKR account'."
    
    async def generate_cnic_response(self, success: bool, user_data: Dict = None, cnic: str = "", 
                                   first_name: str = "", error_reason: str = "", user_message: str = "") -> str:
        """Generate natural CNIC verification responses with enhanced guidance and language support."""
        
        language_prefix = get_language_aware_prompt_prefix(user_message) if user_message else ""
        
        cnic_prompt = f"""{language_prefix}

Generate a natural response for CNIC verification with enhanced user experience.
        
SUCCESS: {success}
USER DATA: {json.dumps(user_data) if user_data else "No data"}
CNIC: {cnic}
FIRST NAME: {first_name}
ERROR REASON: {error_reason}
        
ENHANCED CNIC INSTRUCTIONS:
1. If successful: Welcome them warmly and show their accounts with numbering
2. If failed: Be understanding and help them with correct format
3. Mask account numbers (***-***-1234) for security
4. Be professional but warm
5. Guide them to next steps clearly
6. For format errors, explain the correct format: 12345-1234567-1
7. Show multiple accounts clearly with selection options
        
Generate the response:"""
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=cnic_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating CNIC response: {e}")
            if success and user_data:
                user_language = detect_user_language(user_message) if user_message else "english"
                if user_language == "roman_urdu":
                    return "Swagat hai! Aapke accounts dikh rahe hain lekin display karne mein problem aa rahi hai. Please last 4 digits se select kariye."
                elif user_language == "urdu_script":
                    return "خیر مقدم! آپ کے اکاؤنٹس دکھ رہے ہیں لیکن ڈسپلے کرنے میں مسئلہ ہو رہا ہے۔ براہ کرم آخری 4 ہندسوں سے منتخب کریں۔"
                else:
                    return "Welcome! I can see your accounts but had trouble displaying them. Please select using the last 4 digits of your account number."
            else:
                user_language = detect_user_language(user_message) if user_message else "english"
                if user_language == "roman_urdu":
                    return "Please apna CNIC 12345-1234567-1 format mein dijiye continue karne ke liye."
                elif user_language == "urdu_script":
                    return "براہ کرم اپنا شناختی کارڈ 12345-1234567-1 فارمیٹ میں دیں۔"
                else:
                    return "Please provide your CNIC in the format 12345-1234567-1 to continue."
    
    async def generate_decline_response(self, user_message: str, first_name: str, current_mode: str) -> str:
        """Generate natural decline responses for non-banking queries with enhanced redirection and language support."""
        
        language_prefix = get_language_aware_prompt_prefix(user_message)
        
        decline_prompt = f"""{language_prefix}

Generate a polite decline for a non-banking question with enhanced redirection.
        
USER'S NAME: {first_name}
USER'S MESSAGE: "{user_message}"
CURRENT MODE: {current_mode}
        
ENHANCED DECLINE INSTRUCTIONS:
1. Politely decline to answer non-banking questions
2. Redirect to banking topics naturally and helpfully
3. Be helpful about what you CAN do
4. Don't be rude or robotic
5. Keep it brief but friendly
6. Suggest specific banking topics they might be interested in
7. Maintain the conversational tone
        
Banking topics you can help with:
- Account balance and transactions
- Money transfers
- Bank services and policies
- Account management
        
Generate the decline response:"""
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=decline_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating decline response: {e}")
            user_language = detect_user_language(user_message)
            if user_language == "roman_urdu":
                return f"Main sirf banking questions ka jawab de sakta hoon{', ' + first_name if first_name else ''}. Aapke account, transactions, transfers, ya bank services ke baare mein madad kar sakta hoon."
            elif user_language == "urdu_script":
                return f"میں صرف بینکنگ سوالات کا جواب دے سکتا ہوں{' ' + first_name if first_name else ''}۔ آپ کے اکاؤنٹ، لین دین، ٹرانسفر، یا بینک کی خدمات کے بارے میں مدد کر سکتا ہوں۔"
            else:
                return f"I can only help with banking-related questions{', ' + first_name if first_name else ''}. I can assist you with your account, transactions, transfers, or information about our bank services."
    
    async def generate_mode_switch_response(self, from_mode: str, to_mode: str, user_message: str, 
                                          first_name: str) -> str:
        """Generate natural mode switching responses for enhanced hybrid system with language support."""
        
        language_prefix = get_language_aware_prompt_prefix(user_message)
        
        mode_switch_prompt = f"""{language_prefix}

Generate a natural response for switching between banking modes.
        
USER'S NAME: {first_name}
USER'S MESSAGE: "{user_message}"
FROM MODE: {from_mode}
TO MODE: {to_mode}
        
MODE DESCRIPTIONS:
- rag: General bank information and policies
- account: Personal account access and transactions
- initial: First-time interaction
        
ENHANCED MODE SWITCH INSTRUCTIONS:
1. Acknowledge the switch naturally without being technical
2. Confirm what they're now asking about
3. Be helpful and seamless in the transition
4. Don't mention "switching modes" - just respond to their request
5. Make it feel like a natural conversation flow
        
Generate the transition response:"""
        
        try:
            response = await self.llm.ainvoke([SystemMessage(content=mode_switch_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating mode switch response: {e}")
            user_language = detect_user_language(user_message)
            if user_language == "roman_urdu":
                return f"Samaj gaya{', ' + first_name if first_name else ''}. Main aapki madad karta hoon."
            elif user_language == "urdu_script":
                return f"سمجھ گیا{' ' + first_name if first_name else ''}۔ میں آپ کی مدد کرتا ہوں۔"
            else:
                return f"I understand you'd like help with that{', ' + first_name if first_name else ''}. Let me assist you."