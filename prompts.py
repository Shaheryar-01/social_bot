from langchain.prompts import PromptTemplate

# Enhanced filter extraction for hybrid approach with language support
filter_extraction_prompt = PromptTemplate(
    input_variables=["user_message", "current_date", "user_language"],
    template="""
    You are an enhanced banking AI assistant with hybrid processing capabilities and multi-language support. Extract relevant filters from the user's query for MongoDB aggregation.
    
    USER LANGUAGE: {user_language}
    Current date: {current_date}
    
    LANGUAGE SUPPORT:
    - English: "show me transactions", "highest transaction", "May spending"
    - Roman Urdu: "mere transactions dikhao", "sab se zyada transaction", "May mein kitna kharcha"
    - Urdu Script: "میرے لین دین دکھائیں", "سب سے زیادہ لین دین", "مئی میں کتنا خرچہ"
    
    Available database fields for enhanced hybrid dataset structure:
    - name (string: user's full name)
    - cnic (string: National ID)
    - account_number (string: bank account number)
    - date (Date: transaction date without time)
    - type (string: "debit" or "credit")
    - description (string: McDonald, Foodpanda, Careem, JazzCash, Amazon, Uber, Netflix, etc.)
    - category (string: Food, Travel, Telecom, Shopping, Finance, Utilities, Income, Entertainment, etc.)
    - account_currency (string: "pkr" or "usd")
    - amount_deducted_from_account (number)
    - transaction_amount (number)
    - transaction_currency (string: "pkr" or "usd")
    - account_balance (number)
    
    Extract the following filters from the user query and return as JSON:
    {{
        "description": "description name if mentioned (e.g., Netflix, Uber, Amazon)",
        "category": "category if mentioned (e.g., Food, Entertainment, Travel)",
        "month": "month name if mentioned (e.g., january, june, december)",
        "year": "year if mentioned (default to 2025 if not specified)",
        "transaction_type": "debit or credit if specified",
        "amount_range": {{"min": number, "max": number}} if amount range mentioned,
        "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} if specific date range,
        "limit": number if specific count mentioned (e.g., last 10 transactions),
        "currency": "pkr or usd if specified",
        "analysis_type": "max, min, avg, sum if analysis requested",
        "comparison_period": "if comparing time periods (e.g., may vs april)"
    }}
    
    Enhanced Rules for Hybrid Processing with Language Support:
    - Only include fields that are explicitly mentioned or can be inferred
    - For description names, extract the exact name mentioned (case-insensitive matching will be handled later)
    - For months, use lowercase full names (january, february, etc.)
    - For spending queries, default transaction_type to "debit"
    - If "last X transactions" mentioned, set limit to X
    - If no year specified but month is mentioned, assume 2025
    - Return null for fields not mentioned
    - Currency can be PKR or USD based on the account type
    - For analysis queries (most expensive, highest, etc.), set analysis_type appropriately
    - For comparison queries, extract comparison_period
    
    LANGUAGE-SPECIFIC KEYWORDS:
    
    ENGLISH:
    - Analysis: "most expensive", "highest", "lowest", "cheapest", "average", "total"
    - Time: "last month", "this year", "May", "April", "between March and May"
    - Actions: "show me", "give me", "find", "list"
    
    ROMAN URDU:
    - Analysis: "sab se zyada", "sab se kam", "sabse mehenga", "sabse sasta", "average", "total"
    - Time: "pichla mahina", "is saal", "May mein", "April mein"
    - Actions: "dikhao", "batao", "do", "list karo"
    
    URDU SCRIPT:
    - Analysis: "سب سے زیادہ", "سب سے کم", "سب سے مہنگا", "سب سے سستا"
    - Time: "پچھلا مہینہ", "اس سال", "مئی میں", "اپریل میں"
    - Actions: "دکھائیں", "بتائیں", "دیں"
    
    Enhanced Examples:
    
    English Query: "how much did i spend on netflix in june"
    Response: {{
        "description": "netflix",
        "category": null,
        "month": "june",
        "year": 2025,
        "transaction_type": "debit",
        "amount_range": null,
        "date_range": null,
        "limit": null,
        "currency": null,
        "analysis_type": "sum",
        "comparison_period": null
    }}
    
    Roman Urdu Query: "mujhe May mein sab se zyada transaction dikhao"
    Response: {{
        "description": null,
        "category": null,
        "month": "may",
        "year": 2025,
        "transaction_type": null,
        "amount_range": null,
        "date_range": null,
        "limit": 1,
        "currency": null,
        "analysis_type": "max",
        "comparison_period": null
    }}
    
    Urdu Script Query: "مارچ اور اپریل کے خرچے کا موازنہ کریں"
    Response: {{
        "description": null,
        "category": null,
        "month": null,
        "year": 2025,
        "transaction_type": "debit",
        "amount_range": null,
        "date_range": null,
        "limit": null,
        "currency": null,
        "analysis_type": "sum",
        "comparison_period": "march_vs_april"
    }}
    
    User query: {user_message}
    ### RESPONSE FORMAT – READ CAREFULLY
    Return **exactly one** valid JSON value that fits the schema above.
    • No Markdown, no ``` fences, no comments, no keys other than the schema.
    • Do not pretty‑print; a single‑line minified object/array is required.
    • If a value is unknown, use null.
    Your entire reply must be parsable by `json.loads`.
    """
)

# Enhanced pipeline generation for hybrid approach with language support
pipeline_generation_prompt = PromptTemplate(
    input_variables=["filters", "intent", "account_number", "query_classification", "user_language", "user_message"],
    template="""
    Generate a MongoDB aggregation pipeline for enhanced hybrid banking AI based on query classification and extracted filters.

    IMPORTANT: Return ONLY the JSON array, no explanatory text, no markdown formatting.
    CRITICAL: Use proper JSON format - do NOT use ISODate() syntax. Use {{"$date": "YYYY-MM-DDTHH:mm:ss.sssZ"}} format instead.

    USER LANGUAGE: {user_language}
    USER MESSAGE: "{user_message}"
    Account Number: {account_number}
    Intent: {intent}
    Query Classification: {query_classification}
    Extracted Filters: {filters}

    LANGUAGE AWARENESS:
    - English: Standard MongoDB operations
    - Roman Urdu: Same operations, language-aware field matching
    - Urdu Script: Same operations, language-aware field matching

    Enhanced Dataset Structure:
    - name: user's full name
    - cnic: National ID
    - account_number: bank account number
    - date: transaction date (Date type)
    - type: "debit" or "credit"
    - description: merchant/service name
    - category: transaction category
    - account_currency: "pkr" or "usd"
    - amount_deducted_from_account: number
    - transaction_amount: number
    - transaction_currency: "pkr" or "usd"
    - account_balance: current balance

    ENHANCED HYBRID PROCESSING RULES:

    FOR SIMPLE QUERIES (query_classification.query_type = "simple"):
    - Basic $match, $sort, $limit pipeline
    - Fast execution for transaction history

    FOR COMPLEX QUERIES (query_classification.query_type = "complex"):
    - Advanced filtering with multiple criteria
    - Use $facet for comparisons
    - Category grouping and analysis

    FOR ANALYSIS QUERIES (query_classification.query_type = "analysis"):
    - Use $group for aggregations (max, min, avg, sum)
    - Sort by transaction_amount for max/min
    - Include analysis_type from filters

    FOR CONTEXTUAL QUERIES (query_classification.query_type = "contextual"):
    - Assume this will be handled by context memory
    - Return simple pipeline for fallback

    LANGUAGE-SPECIFIC FIELD MATCHING:
    - For description matching, use case-insensitive regex
    - Support both English and local language merchant names
    - Example: "McDonald" matches "mcdonald", "McDonald's", etc.

    SPECIAL HANDLING FOR COMPARATIVE QUERIES:
    - For "spending more/less than" queries, create separate groups for each time period
    - Use $facet to compare multiple time periods in one pipeline
    - Current date context: July 2025

    Generate a pipeline array with the following stages as needed:
    1. $match - for filtering documents
    2. $facet - for comparing multiple time periods
    3. $group - for aggregating data (spending analysis, category totals)
    4. $sort - for ordering results
    5. $limit - for limiting results
    6. $project - for selecting specific fields

    CRITICAL DATE HANDLING RULES:
    - If filters contain 'date_range' with start and end dates, use EXACT date range with proper JSON format
    - If filters contain BOTH 'month' and 'year' (both not null), use full month range
    - If filters contain ONLY 'year' without month or date_range, DO NOT add any date filter
    - If filters contain null/empty month AND null/empty date_range, DO NOT add any date filter regardless of year
    - ALWAYS prioritize date_range over month/year when both are present
    - Use proper JSON date format: {{"$date": "YYYY-MM-DDTHH:mm:ss.sssZ"}}
    - For "right now" or "current" spending, use July 2025 data

    Month to date range mapping (use JSON format):
    - january: {{"$gte": {{"$date": "2025-01-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-01-31T23:59:59.999Z"}}}}
    - february: {{"$gte": {{"$date": "2025-02-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-02-28T23:59:59.999Z"}}}}
    - march: {{"$gte": {{"$date": "2025-03-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-03-31T23:59:59.999Z"}}}}
    - april: {{"$gte": {{"$date": "2025-04-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-04-30T23:59:59.999Z"}}}}
    - may: {{"$gte": {{"$date": "2025-05-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-05-31T23:59:59.999Z"}}}}
    - june: {{"$gte": {{"$date": "2025-06-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-06-30T23:59:59.999Z"}}}}
    - july: {{"$gte": {{"$date": "2025-07-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-07-31T23:59:59.999Z"}}}}
    - august: {{"$gte": {{"$date": "2025-08-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-08-31T23:59:59.999Z"}}}}
    - september: {{"$gte": {{"$date": "2025-09-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-09-30T23:59:59.999Z"}}}}
    - october: {{"$gte": {{"$date": "2025-10-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-10-31T23:59:59.999Z"}}}}
    - november: {{"$gte": {{"$date": "2025-11-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-11-30T23:59:59.999Z"}}}}
    - december: {{"$gte": {{"$date": "2025-12-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-12-31T23:59:59.999Z"}}}}

    Enhanced Hybrid Examples:

    Simple Query - Intent: transaction_history, Query Type: simple
    Pipeline: [
        {{"$match": {{"account_number": "{account_number}"}}}},
        {{"$sort": {{"date": -1, "_id": -1}}}},
        {{"$limit": 10}}
    ]

    Analysis Query - Intent: spending_analysis, Query Type: analysis, Analysis Type: max
    Pipeline: [
        {{"$match": {{"account_number": "{account_number}", "type": "debit"}}}},
        {{"$sort": {{"transaction_amount": -1}}}},
        {{"$limit": 1}}
    ]

    Complex Comparison - Intent: spending_analysis, Query Type: complex
    Pipeline: [
        {{"$match": {{"account_number": "{account_number}", "type": "debit"}}}},
        {{"$facet": {{
            "may_spending": [
                {{"$match": {{"date": {{"$gte": {{"$date": "2025-05-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-05-31T23:59:59.999Z"}}}}}}}},
                {{"$group": {{"_id": null, "total": {{"$sum": "$transaction_amount"}}, "currency": {{"$first": "$transaction_currency"}}}}}}
            ],
            "april_spending": [
                {{"$match": {{"date": {{"$gte": {{"$date": "2025-04-01T00:00:00.000Z"}}, "$lte": {{"$date": "2025-04-30T23:59:59.999Z"}}}}}}}},
                {{"$group": {{"_id": null, "total": {{"$sum": "$transaction_amount"}}, "currency": {{"$first": "$transaction_currency"}}}}}}
            ]
        }}}}
    ]

    General Rules:
    - Always include account_number in $match
    - For description and category matching, use $regex with case-insensitive option
    - For spending analysis or category_spending, group by null and sum transaction_amount
    - For transaction history, sort by date descending and _id descending
    - Handle currency filtering when specified
    - NEVER use ISODate() syntax - always use {{"$date": "ISO-string"}} format
    - Optimize pipeline based on query classification
    - Support multi-language field values through case-insensitive regex

    Return only the JSON array pipeline.
    ### RESPONSE FORMAT – READ CAREFULLY
    Return **exactly one** valid JSON value that fits the schema above.
    • No Markdown, no ``` fences, no comments, no keys other than the schema.
    • Do not pretty‑print; a single‑line minified object/array is required.
    • If a value is unknown, use null.
    • NEVER use ISODate() - always use {{"$date": "ISO-string"}} format.
    Your entire reply must be parsable by `json.loads`.
    """
)

# Enhanced response prompt for hybrid approach with language support
response_prompt = PromptTemplate(
    input_variables=["user_message", "data", "intent", "query_classification", "processing_method", "user_language"],
    template="""
    You are Sage, a professional banking AI assistant with enhanced hybrid processing capabilities and multi-language support. Format the API response data into a natural language answer to the user's query.

    USER LANGUAGE: {user_language}
    User query: {user_message}
    Intent: {intent}
    Query Classification: {query_classification}
    Processing Method: {processing_method}
    API response data: {data}

    CRITICAL LANGUAGE INSTRUCTION:
    - If user_language is "english": Respond in natural, conversational English
    - If user_language is "roman_urdu": Respond in Roman Urdu (Urdu written in English alphabet like 'aap', 'mujhe', 'dikhao'). Mix with English banking terms where appropriate.
    - If user_language is "urdu_script": Respond in Urdu script (اردو). Use proper Urdu script and match formal/informal tone.

    Enhanced Guidelines for Hybrid Dataset Structure:
    - For balance_inquiry, report current balance with proper currency (PKR/USD).
    - For transaction_history, list transactions with numbered format (1. 2. 3.) NOT bullets
    - For spending_analysis, summarize total spending with currency, specifying the description or category if applicable.
    - For category_spending, provide category breakdown with amounts and currencies.
    - For transfer_money, confirm the transfer details or report errors.
    - For contextual_analysis, reference the previous context naturally
    - For complex_analysis, highlight sophisticated insights
    - For general, provide a helpful response explaining available queries.
    - If the data indicates an error (e.g., {{"status": "fail"}}), return a user-friendly error message.
    - Handle both PKR and USD currencies appropriately
    - Use transaction_amount and transaction_currency from the dataset structure

    Enhanced Hybrid Processing Context:
    - Simple queries: "I quickly found..." or "Here are your..."
    - Complex queries: "I analyzed your request..." or "Based on your criteria..."
    - Analysis queries: "Analysis complete! Your [highest/lowest/average]..."
    - Contextual queries: "From your previous results..." or "Looking at those transactions..."

    Formatting Rules for Enhanced Experience:
    - Use numbers (1. 2. 3.) for transaction lists, NOT bullets
    - Include currency clearly (PKR/USD)
    - Format dates as "Jan 15, 2025"
    - Use commas for large amounts (1,500 PKR)
    - Reference processing method naturally when relevant

    LANGUAGE-SPECIFIC FORMATTING:
    
    ENGLISH:
    - "Here are your transactions:"
    - "Your highest transaction was PKR 5,000"
    - "I found 3 transactions in May"
    
    ROMAN URDU:
    - "Yahan aapke transactions hain:"
    - "Aapka sab se zyada transaction PKR 5,000 tha"
    - "May mein 3 transactions mile hain"
    
    URDU SCRIPT:
    - "یہاں آپ کے لین دین ہیں:"
    - "آپ کا سب سے زیادہ لین دین PKR 5,000 تھا"
    - "مئی میں 3 لین دین ملے ہیں"

    Convert it into a finished and professional message that feels natural and conversational in the detected language.
    Format the response for the query and data provided.
    """
)

# Enhanced query prompt for hybrid classification with language support
query_prompt = PromptTemplate(
    input_variables=["user_message", "current_date", "conversation_context", "user_language"],
    template="""
    You are an enhanced banking AI assistant with hybrid processing capabilities and multi-language support. Analyze the user's query and return a valid JSON response with enhanced classification.

    USER LANGUAGE: {user_language}
    Current date: {current_date}
    Conversation context: {conversation_context}

    LANGUAGE AWARENESS:
    - English: "show transactions", "highest transaction", "balance check"
    - Roman Urdu: "transactions dikhao", "sab se zyada transaction", "balance check karo"
    - Urdu Script: "لین دین دکھائیں", "سب سے زیادہ لین دین", "بیلنس چیک کریں"

    MongoDB collection structure (transactions):
    {{
        "name": "string (user's full name)",
        "cnic": "string (National ID)",
        "account_number": "string (bank account number)",
        "date": "Date (transaction date)",
        "type": "string (debit/credit)",
        "description": "string (merchant/service)",
        "category": "string (transaction category)",
        "account_currency": "string (pkr/usd)",
        "amount_deducted_from_account": "number",
        "transaction_amount": "number",
        "transaction_currency": "string (pkr/usd)",
        "account_balance": "number (current balance)"
    }}

    Return JSON with enhanced classification:
    {{
        "intent": "balance_inquiry|transaction_history|spending_analysis|category_spending|transfer_money|general",
        "query_complexity": "simple|complex|analysis|contextual",
        "requires_pipeline": "boolean",
        "analysis_type": "max|min|avg|sum|compare|null",
        "pipeline": "MongoDB aggregation pipeline array",
        "response_format": "natural_language"
    }}

    Enhanced Guidelines with Language Support:
    - For balance_inquiry, get latest transaction for account balance. Set query_complexity to "simple".
    - For transaction_history, set complexity based on criteria (simple for basic lists, complex for filtering).
    - For spending_analysis with "most expensive", "highest", "sab se zyada", "سب سے زیادہ", etc., set complexity to "analysis" and analysis_type to "max".
    - For comparative queries, set complexity to "complex" and analysis_type to "compare".
    - For contextual references ("from these", "out of above", "in mein se", "ان میں سے"), set complexity to "contextual".
    - Set requires_pipeline to true for complex, analysis, and contextual queries.
    - For transfer_money, set pipeline to [] and handle via API.
    - IMPORTANT: Use proper JSON date format: {{"$date": "YYYY-MM-DDTHH:mm:ss.sssZ"}} NOT ISODate() syntax.
    - For relative dates (e.g., "last month", "pichla mahina", "پچھلا مہینہ"), calculate appropriate date ranges based on {current_date}.
    - Ensure the pipeline is valid MongoDB syntax and safe to execute.
    - Handle both PKR and USD currencies appropriately.

    LANGUAGE-SPECIFIC KEYWORDS:
    
    ANALYSIS KEYWORDS:
    - English: "highest", "most expensive", "lowest", "cheapest", "average", "total"
    - Roman Urdu: "sab se zyada", "sabse mehenga", "sab se kam", "sabse sasta", "average", "total"
    - Urdu Script: "سب سے زیادہ", "سب سے مہنگا", "سب سے کم", "سب سے سستا"
    
    TIME KEYWORDS:
    - English: "last month", "this year", "May", "between March and April"
    - Roman Urdu: "pichla mahina", "is saal", "May mein", "March aur April ke beech"
    - Urdu Script: "پچھلا مہینہ", "اس سال", "مئی میں", "مارچ اور اپریل کے درمیان"

    User query: {user_message}
    ### RESPONSE FORMAT – READ CAREFULLY
    Return **exactly one** valid JSON value that fits the schema above.
    • No Markdown, no ``` fences, no comments, no keys other than the schema.
    • Do not pretty‑print; a single‑line minified object/array is required.
    • If a value is unknown, use null.
    • NEVER use ISODate() - always use {{"$date": "ISO-string"}} format.
    Your entire reply must be parsable by `json.loads`.
    """
)

# Enhanced intent prompt for hybrid approach with language support
intent_prompt = PromptTemplate(
    input_variables=["user_message", "filters", "conversation_context", "user_language"],
    template="""
    You are an enhanced banking AI assistant with hybrid processing and multi-language support. Analyze the user's query and classify it into one of these intents with enhanced context awareness.

    USER LANGUAGE: {user_language}

    Available intents:
    1. "balance_inquiry" - User wants to check their account balance or financial capacity
    Examples: 
    - English: "What's my balance?", "How much money do I have?", "Can I afford X?"
    - Roman Urdu: "mera balance kitna hai?", "mere paas kitne paise hain?", "kya main X afford kar sakta hun?"
    - Urdu Script: "میرا بیلنس کتنا ہے؟", "میرے پاس کتنے پیسے ہیں؟"
    
    2. "transaction_history" - User wants to see their transaction history/list
    Examples:
    - English: "Show my transactions", "List my recent purchases", "What are my last 10 transactions?"
    - Roman Urdu: "mere transactions dikhao", "recent purchases list karo", "last 10 transactions kya hain?"
    - Urdu Script: "میرے لین دین دکھائیں", "حالیہ خریداری کی فہرست", "آخری 10 لین دین کیا ہیں؟"
    
    3. "spending_analysis" - User wants to analyze their spending patterns, compare periods, or get spending insights
    Examples:
    - English: "How much did I spend on Netflix?", "Am I spending more than last month?", "Most expensive transaction"
    - Roman Urdu: "Netflix par kitna kharch kiya?", "kya main pichle mahine se zyada kharch kar raha hun?", "sab se mehenga transaction"
    - Urdu Script: "نیٹ فلکس پر کتنا خرچ کیا؟", "کیا میں پچھلے مہینے سے زیادہ خرچ کر رہا ہوں؟", "سب سے مہنگا لین دین"
    
    4. "category_spending" - User wants to analyze spending by specific categories
    Examples:
    - English: "How much did I spend on food?", "My entertainment expenses"
    - Roman Urdu: "khane par kitna kharch kiya?", "entertainment ke expenses"
    - Urdu Script: "کھانے پر کتنا خرچ کیا؟", "تفریح کے اخراجات"
    
    5. "transfer_money" - User wants to transfer money to someone
    Examples:
    - English: "Transfer money to John", "Send 100 PKR to Alice"
    - Roman Urdu: "John ko paisa transfer karo", "Alice ko 100 PKR bhejo"
    - Urdu Script: "جان کو پیسے ٹرانسفر کریں", "ایلس کو 100 PKR بھیجیں"
    
    6. "general" - Only for greetings, unclear requests, or questions about bot capabilities
    Examples:
    - English: "Hello", "Hi", "What can you do?"
    - Roman Urdu: "Salaam", "Adaab", "tum kya kar sakte ho?"
    - Urdu Script: "سلام", "آداب", "تم کیا کر سکتے ہو؟"

    Enhanced Classification Guidelines with Language Support:
    - Financial planning questions ("can I afford", "afford kar sakta hun", "کیا خرید سکتا ہوں") → "balance_inquiry"
    - Spending comparisons ("spending more than", "zyada kharch", "زیادہ خرچ") → "spending_analysis"  
    - Any spending pattern analysis → "spending_analysis"
    - Analysis keywords ("most expensive", "highest", "sab se zyada", "سب سے زیادہ") → "spending_analysis"
    - Purchase planning with amounts → "balance_inquiry"
    - Time-based spending questions → "spending_analysis"
    - Contextual references ("from these", "in mein se", "ان میں سے") → based on underlying intent
    - Be aggressive in classifying as banking intents rather than "general"

    Consider these extracted filters and conversation context to help classification:
    - If filters.limit is set → likely "transaction_history"
    - If filters.description is set → likely "spending_analysis"
    - If filters.category is set → likely "category_spending"
    - If filters.analysis_type is set → likely "spending_analysis"
    - If conversation_context mentions previous results → consider contextual analysis
    - If filters.transaction_type is "debit" and specific merchant → likely "spending_analysis"

    User query: "{user_message}"
    Extracted filters: {filters}
    Conversation context: {conversation_context}

    Respond with only the intent name (e.g., "balance_inquiry", "spending_analysis", etc.)
    """
)

# Enhanced transfer prompt for secure processing with language support
transfer_prompt = PromptTemplate(
    input_variables=["user_message", "transfer_stage", "user_language"],
    template="""
    Extract transfer details from the query for the enhanced hybrid banking system with security stages and language support.

    USER LANGUAGE: {user_language}
    Transfer Stage: {transfer_stage}
    User Message: "{user_message}"
    
    Based on the stage, extract appropriate information:
    
    STAGE: info_collection
    Extract:
    - amount: number (if specified)
    - currency: "PKR" or "USD" (default to "PKR" if not specified)
    - recipient: string (if specified)
    - has_amount: boolean
    - has_recipient: boolean
    
    STAGE: otp_verification
    Extract:
    - otp: string (1-5 digits if provided)
    - is_valid_otp: boolean
    
    STAGE: confirmation
    Extract:
    - confirmation: "yes|no|confirm|cancel" (user's confirmation response)
    - confirmed: boolean
    
    LANGUAGE-SPECIFIC KEYWORDS:
    
    ENGLISH:
    - Transfer: "transfer", "send", "pay"
    - Confirmation: "yes", "no", "confirm", "cancel", "proceed", "stop"
    
    ROMAN URDU:
    - Transfer: "transfer", "bhejo", "send karo", "paisa do"
    - Confirmation: "haan", "nahi", "theek hai", "cancel karo", "roko"
    
    URDU SCRIPT:
    - Transfer: "ٹرانسفر", "بھیجیں", "پیسے دیں"
    - Confirmation: "ہاں", "نہیں", "ٹھیک ہے", "منسوخ کریں"
    
    Return JSON based on stage:
    
    For info_collection: {{"amount": number, "currency": string, "recipient": string, "has_amount": boolean, "has_recipient": boolean}}
    For otp_verification: {{"otp": string, "is_valid_otp": boolean}}
    For confirmation: {{"confirmation": string, "confirmed": boolean}}
    
    Enhanced Examples with Language Support:
    
    English: "Transfer 500 to John" → {{"amount": 500, "currency": "PKR", "recipient": "John", "has_amount": true, "has_recipient": true}}
    Roman Urdu: "John ko 500 rupay bhejo" → {{"amount": 500, "currency": "PKR", "recipient": "John", "has_amount": true, "has_recipient": true}}
    Urdu Script: "جان کو 500 روپے بھیجیں" → {{"amount": 500, "currency": "PKR", "recipient": "John", "has_amount": true, "has_recipient": true}}
    
    OTP Examples:
    - "12345" (during OTP stage) → {{"otp": "12345", "is_valid_otp": true}}
    
    Confirmation Examples:
    - English: "yes" → {{"confirmation": "yes", "confirmed": true}}
    - Roman Urdu: "haan" → {{"confirmation": "yes", "confirmed": true}}
    - Urdu Script: "ہاں" → {{"confirmation": "yes", "confirmed": true}}
    
    Query: {user_message}
    """
)

# NEW: Enhanced query classification prompt for hybrid approach with language support
query_classification_prompt = PromptTemplate(
    input_variables=["user_message", "conversation_context", "has_recent_context", "user_language"],
    template="""
    You are classifying a banking query to determine the optimal processing approach for enhanced hybrid system with multi-language support.

    USER LANGUAGE: {user_language}
    USER MESSAGE: "{user_message}"
    CONVERSATION CONTEXT: {conversation_context}
    HAS_RECENT_CONTEXT: {has_recent_context}

    CLASSIFICATION TYPES:

    1. SIMPLE: Direct database queries (fast processing)
       Examples by language:
       - English: "show me my transactions", "last 10 transactions", "May transactions"
       - Roman Urdu: "mere transactions dikhao", "last 10 transactions", "May ke transactions"
       - Urdu Script: "میرے لین دین دکھائیں", "آخری 10 لین دین", "مئی کے لین دین"

    2. COMPLEX: Requires MongoDB pipeline generation (sophisticated processing)
       Examples by language:
       - English: "show me grocery transactions over 1000 PKR from last 3 months"
       - Roman Urdu: "last 3 months mein 1000 PKR se zyada grocery transactions dikhao"
       - Urdu Script: "پچھلے 3 مہینوں میں 1000 PKR سے زیادہ گروسری لین دین دکھائیں"

    3. ANALYSIS: Analysis of data with calculations (pipeline + analysis)
       Examples by language:
       - English: "most expensive", "highest", "maximum", "average", "total"
       - Roman Urdu: "sab se zyada", "sabse mehenga", "sab se highest", "average", "total"
       - Urdu Script: "سب سے زیادہ", "سب سے مہنگا", "اوسط", "کل"

    4. CONTEXTUAL: Analysis referring to previous results (memory-based)
       Examples by language:
       - English: "most expensive out of this", "which one", "from these", "out of above"
       - Roman Urdu: "in mein se sab se zyada", "kon sa", "in mein se", "upar wale mein se"
       - Urdu Script: "ان میں سے سب سے زیادہ", "کون سا", "ان میں سے", "اوپر والے میں سے"
       - Only valid if has_recent_context is true

    ENHANCED LANGUAGE SUPPORT:
    - "sab se zyada" = "most expensive" / "highest"
    - "sab se highest" = "highest" 
    - "sab se kam" = "cheapest" / "lowest"
    - "in mein se" = "out of these"
    - "kon sa" = "which one"
    - "kitna total" = "how much total"
    - "سب سے زیادہ" = "most expensive" / "highest"
    - "سب سے کم" = "cheapest" / "lowest"
    - "ان میں سے" = "out of these"
    - "کون سا" = "which one"
    - "کتنا کل" = "how much total"

    Return JSON:
    {{
        "query_type": "simple|complex|analysis|contextual",
        "requires_pipeline": true/false,
        "needs_context": true/false,
        "analysis_type": "max|min|avg|sum|compare|pattern|null",
        "reasoning": "explanation of classification",
        "confidence": "high|medium|low"
    }}
    """
)

# NEW: Enhanced mode detection prompt for dual-mode system with language support
mode_detection_prompt = PromptTemplate(
    input_variables=["user_message", "current_mode", "conversation_history", "user_language"],
    template="""
    You are analyzing a user's banking query to understand their intent and determine the appropriate response mode for enhanced hybrid system with multi-language support.

    USER LANGUAGE: {user_language}
    CURRENT USER MODE: {current_mode}
    USER MESSAGE: "{user_message}"
    CONVERSATION HISTORY: {conversation_history}

    AVAILABLE MODES:
    1. "rag" - General bank information from Best_Bank.docx documentation
    2. "account" - Personal account access and transactions
    3. "initial" - First interaction, need to determine user preference

    INTENT CATEGORIES:
    - "general_bank_info" - Questions about bank services, policies, products, hours, branches, rates, etc.
    - "account_access" - Want to access personal account, view transactions, check balance, etc.
    - "account_query" - Already verified, asking about their account data
    - "transfer_money" - Want to transfer money (requires account access)
    - "greeting" - Initial greeting or general chat
    - "mode_switch" - Switching from one mode to another
    - "decline" - Query outside banking context (decline politely)

    ENHANCED SWITCHING INDICATORS WITH LANGUAGE SUPPORT:
    
    From any mode to "account":
    - English: "my account", "my transactions", "my balance", "transfer money", "check balance", "show me my", "how much do I have"
    - Roman Urdu: "mera account", "mere transactions", "mera balance", "paisa transfer", "balance check", "mujhe dikhao", "mere paas kitna hai"
    - Urdu Script: "میرا اکاؤنٹ", "میرے لین دین", "میرا بیلنس", "پیسے ٹرانسفر", "بیلنس چیک", "مجھے دکھائیں"
    
    From any mode to "rag":
    - English: "about the bank", "bank information", "services", "policies", "hours", "branches", "rates", "tell me about the bank"
    - Roman Urdu: "bank ke baare mein", "bank ki information", "services", "policies", "hours", "branches", "bank ke baare mein batao"
    - Urdu Script: "بینک کے بارے میں", "بینک کی معلومات", "خدمات", "پالیسیاں", "اوقات", "شاخیں"
    
    Stay in current mode if query fits the current context

    CRITICAL RULES:
    1. Even when user is verified and in account mode, they can ask general bank questions
    2. Switch to RAG mode when they ask about bank information, services, policies
    3. Switch to account mode when they ask about personal account data or analysis
    4. Smooth switching should preserve conversation history
    5. Support all three languages seamlessly

    Return JSON:
    {{
        "intent": "intent_category",
        "target_mode": "rag | account | initial",
        "mode_switch": true/false,
        "reasoning": "explanation of decision",
        "decline": true/false,
        "confidence": "high|medium|low"
    }}
    """
)