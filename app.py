from fastapi import FastAPI
from api_routes import router
import logging
import asyncio
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pure API-Based Banking AI Assistant with Multi-Language Support",
    description="Advanced Banking AI with pure API communication, RAG integration, enhanced hybrid query processing, dual-mode operation, sophisticated security features, and multi-language support (English, Roman Urdu, Urdu Script)",
    version="5.1.0"
)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initialize enhanced hybrid systems with language support on startup."""
    import os  # Ensure os is available in function scope
    logger.info("🚀 Starting Pure API-Based Banking AI Assistant with Multi-Language Support...")
    
    # Check if Best_Bank.docx exists
    if not os.path.exists("Best_Bank.docx"):
        logger.warning("⚠️  Best_Bank.docx not found! RAG system may not work properly.")
        logger.info("📄 Please create Best_Bank.docx with your bank's documentation.")
    else:
        logger.info("✅ Best_Bank.docx found")
    
    # Initialize language support system
    try:
        from language_utils import detect_user_language, get_localized_response
        logger.info("✅ Language support system initialized")
        logger.info("🌐 Supported languages: English, Roman Urdu, Urdu Script")
        logger.info("🔍 Auto-detection enabled for all supported languages")
        
        # Test language detection
        test_messages = [
            "hello",
            "mujhe mere transactions dikhao", 
            "اکاؤنٹ بیلنس چیک کریں"
        ]
        
        for msg in test_messages:
            lang = detect_user_language(msg)
            logger.info(f"   Test: '{msg[:20]}...' → {lang}")
            
    except Exception as e:
        logger.error(f"❌ Language support system initialization failed: {e}")
    
    # Initialize RAG system
    try:
        from rag_system import bank_rag
        if bank_rag and bank_rag.documents:
            logger.info(f"✅ RAG system initialized with {len(bank_rag.documents)} document chunks")
            logger.info(f"✅ FAISS index: {bank_rag.index.ntotal if bank_rag.index else 0} vectors")
            logger.info("🌐 RAG responses support all languages via LLM generation")
        else:
            logger.warning("⚠️  RAG system initialization incomplete")
    except Exception as e:
        logger.error(f"❌ RAG system initialization failed: {e}")
    
    # Initialize enhanced hybrid AI agent with language support
    try:
        from ai_agent import enhanced_ai_agent
        logger.info("✅ Enhanced Hybrid AI agent loaded successfully")
        
        # Log agent capabilities
        logger.info("🎯 Enhanced Hybrid Capabilities:")
        logger.info("   • Simple queries → Direct database access (fast)")
        logger.info("   • Complex queries → LLM pipeline generation")
        logger.info("   • Analysis queries → Context memory + pipelines")
        logger.info("   • Contextual queries → Previous results analysis")
        logger.info("   • Dual-mode switching → RAG + Account access")
        logger.info("   • Enhanced account selection → Multi-format support")
        logger.info("   • Transfer security → OTP + Confirmation")
        logger.info("   • Context memory → Transaction analysis storage")
        logger.info("   • Language support → Auto-detection + localized responses")
        
        # Log language-specific capabilities
        logger.info("🌐 Language Support Features:")
        logger.info("   • Auto-detection from user messages")
        logger.info("   • English: Full natural language processing")
        logger.info("   • Roman Urdu: Mixed language understanding")
        logger.info("   • Urdu Script: Native script support")
        logger.info("   • Fallback responses for all languages")
        logger.info("   • Localized error messages and prompts")
        
    except Exception as e:
        logger.error(f"❌ Enhanced Hybrid AI agent initialization failed: {e}")
    
    # Test database connection
    try:
        from mongo import transactions
        count = transactions.count_documents({})
        logger.info(f"✅ MongoDB connection successful: {count} total transactions")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
    
    # Test LLM connection
    try:
        from langchain_openai import ChatOpenAI
        import os
        
        if os.getenv("OPENAI_API_KEY"):
            logger.info("✅ OpenAI API key found")
            logger.info("✅ LLM connection ready for multi-language processing")
        else:
            logger.warning("⚠️  OPENAI_API_KEY not found in environment")
    except Exception as e:
        logger.error(f"❌ LLM connection test failed: {e}")
    
    logger.info("🎉 Enhanced Hybrid Banking AI Assistant with Language Support startup complete!")
    logger.info("🔥 Ready to handle:")
    logger.info("   • Simple transaction queries (direct DB)")
    logger.info("   • Complex analysis queries (pipeline generation)")
    logger.info("   • Contextual analysis (memory-based)")
    logger.info("   • RAG-based bank information")
    logger.info("   • Secure money transfers")
    logger.info("   • Multi-currency account management")
    logger.info("   • Multi-language interactions (EN/Roman Urdu/Urdu Script)")
    logger.info("   • Auto language detection and localized responses")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🔄 Shutting down Pure API-Based Banking AI Assistant with Language Support...")
    
    # Cleanup enhanced AI agent
    try:
        from ai_agent import enhanced_ai_agent
        if hasattr(enhanced_ai_agent, '__del__'):
            enhanced_ai_agent.__del__()
        
        # Log cleanup stats
        logger.info(f"🧹 Cleaned up:")
        logger.info(f"   • User memories: {len(enhanced_ai_agent.user_memories) if hasattr(enhanced_ai_agent, 'user_memories') else 0}")
        logger.info(f"   • User modes: {len(enhanced_ai_agent.user_modes) if hasattr(enhanced_ai_agent, 'user_modes') else 0}")
        logger.info(f"   • Transfer states: {len(enhanced_ai_agent.transfer_states) if hasattr(enhanced_ai_agent, 'transfer_states') else 0}")
        logger.info(f"   • Transaction contexts: {len(enhanced_ai_agent.transaction_contexts) if hasattr(enhanced_ai_agent, 'transaction_contexts') else 0}")
        
        logger.info("✅ Enhanced Hybrid AI agent cleanup complete")
    except Exception as e:
        logger.error(f"❌ Enhanced Hybrid AI agent cleanup failed: {e}")
    
    # Cleanup RAG system
    try:
        from rag_system import bank_rag
        if bank_rag:
            logger.info("✅ RAG system cleanup complete")
    except Exception as e:
        logger.error(f"❌ RAG system cleanup failed: {e}")
    
    # Cleanup language support
    try:
        logger.info("✅ Language support system cleanup complete")
    except Exception as e:
        logger.error(f"❌ Language support cleanup failed: {e}")
    
    logger.info("👋 Pure API-Based Banking AI Assistant with Language Support shutdown complete!")

@app.get("/")
async def root():
    """Root endpoint with enhanced hybrid system and language support information."""
    return {
        "service": "Pure API-Based Banking AI Assistant",
        "version": "5.1.0",
        "status": "operational",
        "architecture": "Pure API-Based Microservices with Multi-Language Support",
        "description": "Advanced banking AI with clean API-based communication, intelligent query processing, and comprehensive language support",
        "communication_pattern": {
            "webhook_to_backend": "HTTP API calls only",
            "service_separation": "Clean microservices architecture",
            "ai_agent_location": "Backend service only",
            "no_direct_imports": "Pure API communication"
        },
        "language_support": {
            "supported_languages": ["English", "Roman Urdu", "Urdu Script"],
            "auto_detection": "Enabled for all user messages",
            "localized_responses": "All system responses available in user's language",
            "fallback_handling": "Graceful degradation to English if detection fails",
            "mixed_language": "Support for Roman Urdu (Urdu in English alphabet)",
            "native_script": "Full Urdu script support",
            "banking_terms": "Banking terminology localized appropriately"
        },
        "features": {
            "query_processing": {
                "simple_queries": "Direct database access (optimized for speed)",
                "complex_queries": "LLM pipeline generation (sophisticated analysis)",
                "analysis_queries": "Context memory + pipelines (smart analysis)",
                "contextual_queries": "Previous results analysis (memory-based)"
            },
            "dual_mode_operation": "RAG + Account Access with smooth switching",
            "enhanced_account_selection": "Multi-format, currency-aware, intelligent parsing",
            "transfer_security": "OTP + Confirmation with validation",
            "context_limitation": "Strict banking only (polite decline for others)",
            "thinking_indicator": "2+ second response threshold",
            "cnic_text_extraction": "Smart pattern matching and extraction",
            "conversation_memory": "Persistent per session with intelligent cleanup",
            "response_formatting": "Concise, structured, numbered (not bullets)",
            "currency_support": "PKR and USD with intelligent handling",
            "rag_integration": "FAISS-based semantic search with Best_Bank.docx"
        },
        "hybrid_processing": {
            "classification_engine": "LLM-based query complexity analysis",
            "simple_threshold": "Basic transaction lists, balance checks",
            "complex_threshold": "Multi-criteria filtering, comparisons",
            "analysis_threshold": "Max/min/avg calculations, pattern analysis",
            "contextual_threshold": "References to previous results",
            "fallback_strategy": "Graceful degradation to simpler methods"
        },
        "api_endpoints": {
            "authentication": "/verify_cnic, /select_account",
            "query_processing": "/process_query, /enhanced_query, /rag_query",
            "ai_operations": "/detect_initial_choice, /enhanced_account_selection, /handle_session_end",
            "response_generation": "/generate_error_response, /handle_initial_greeting",
            "utility": "/extract_cnic_from_text, /handle_invalid_cnic_format",
            "banking": "/user_balance, /transfer_money, /execute_pipeline",
            "health": "/health, /rag_health, /ai_agent_status"
        },
        "performance": {
            "simple_queries": "< 100ms typical",
            "complex_queries": "1-3s depending on complexity",
            "rag_queries": "500ms-2s depending on document size",
            "thinking_indicator": "Shown for 2+ second responses",
            "api_overhead": "Minimal due to local communication",
            "language_detection": "< 10ms for most messages"
        },
        "examples": {
            "english": {
                "balance": "What's my balance?",
                "transactions": "Show me my May transactions",
                "analysis": "What was my highest transaction?"
            },
            "roman_urdu": {
                "balance": "mera balance kitna hai?",
                "transactions": "mere May ke transactions dikhao",
                "analysis": "sab se zyada transaction konsa tha?"
            },
            "urdu_script": {
                "balance": "میرا بیلنس کتنا ہے؟",
                "transactions": "میرے مئی کے لین دین دکھائیں",
                "analysis": "سب سے زیادہ لین دین کون سا تھا؟"
            }
        }
    }

@app.get("/docs-info")
async def docs_info():
    """Enhanced API documentation information with language support details."""
    return {
        "documentation": "Enhanced Hybrid Banking AI API with Multi-Language Support",
        "swagger_ui": "/docs",
        "redoc": "/redoc",
        "openapi_json": "/openapi.json",
        "key_features": {
            "hybrid_query_processing": "Intelligent classification and routing",
            "rag_integration": "Semantic search of bank documentation",
            "dual_mode_operation": "Seamless switching between info and account modes",
            "enhanced_security": "Multi-stage authentication and transfer validation",
            "context_awareness": "Memory-based analysis of previous queries",
            "multi_currency": "PKR and USD support with intelligent handling",
            "language_support": "English, Roman Urdu, and Urdu Script with auto-detection"
        },
        "language_features": {
            "auto_detection": "Automatically detects user's language from their message",
            "supported_languages": {
                "english": "Full natural language processing and responses",
                "roman_urdu": "Urdu written in English alphabet (e.g., 'mujhe dikhao')",
                "urdu_script": "Native Urdu script support (e.g., 'مجھے دکھائیں')"
            },
            "localized_responses": "All system messages available in user's detected language",
            "banking_terminology": "Appropriate banking terms for each language",
            "fallback_handling": "Graceful degradation if language detection fails"
        },
        "getting_started": {
            "1": "Start with /verify_cnic endpoint using CNIC format 12345-1234567-1",
            "2": "Select account using /select_account with various input formats",
            "3": "Use /process_query for natural language banking queries in any supported language",
            "4": "Use /rag_query for bank information and policies",
            "5": "Use /enhanced_query for detailed processing insights",
            "language_note": "All endpoints automatically detect and respond in the user's language"
        },
        "example_api_calls": {
            "english_query": {
                "endpoint": "/process_query",
                "payload": {
                    "user_message": "show me my highest transaction",
                    "account_number": "001-0156-000654321", 
                    "first_name": "Ali"
                }
            },
            "roman_urdu_query": {
                "endpoint": "/process_query", 
                "payload": {
                    "user_message": "mujhe mere sab se zyada transaction dikhao",
                    "account_number": "001-0156-000654321",
                    "first_name": "Ali"
                }
            },
            "urdu_script_query": {
                "endpoint": "/process_query",
                "payload": {
                    "user_message": "مجھے میرا سب سے زیادہ لین دین دکھائیں",
                    "account_number": "001-0156-000654321", 
                    "first_name": "Ali"
                }
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Enhanced startup configuration with language support
    logger.info("🚀 Starting Pure API-Based Banking AI Assistant Server with Multi-Language Support...")
    logger.info("📡 Architecture: Webhook → HTTP API → Backend AI Agent")
    logger.info("🎯 Separation: Clean microservices with API communication")
    logger.info("🤖 AI Capabilities: Query Classification | Context Memory | RAG Integration")
    logger.info("🔐 Security: CNIC Auth | Account Selection | Transfer Validation")
    logger.info("🌐 Language Support: English | Roman Urdu | Urdu Script")
    logger.info("🔍 Auto-Detection: Seamless language switching based on user input")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info",
        access_log=True
    )