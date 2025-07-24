# Enhanced state.py with CNIC-based authentication and hybrid approach support
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Authentication states
authenticated_users = {}  # sender_id -> {cnic, name, verification_stage, accounts, mode, etc.}
user_sessions = {} 
processed_messages = set()

# Enhanced verification stages for hybrid approach
VERIFICATION_STAGES = {
    "NOT_VERIFIED": "not_verified",
    "CNIC_VERIFIED": "cnic_verified", 
    "ACCOUNT_SELECTED": "account_selected"
}

# Enhanced user modes for hybrid system
USER_MODES = {
    "INITIAL": "initial",
    "RAG": "rag",           # General bank information mode
    "ACCOUNT": "account",   # Personal account access mode
    "TRANSFER": "transfer", # Money transfer mode
    "ANALYSIS": "analysis"  # Transaction analysis mode
}

# Enhanced session data structure
class EnhancedUserSession:
    """Enhanced user session with hybrid approach support."""
    
    def __init__(self, sender_id: str):
        self.sender_id = sender_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.verification_stage = VERIFICATION_STAGES["NOT_VERIFIED"]
        self.mode = USER_MODES["INITIAL"]
        self.cnic = None
        self.name = None
        self.accounts = []
        self.selected_account = None
        self.conversation_count = 0
        self.mode_switches = 0
        self.last_query_type = None
        self.context_data = {}
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
        self.conversation_count += 1
    
    def switch_mode(self, new_mode: str):
        """Switch user mode and track switches."""
        if self.mode != new_mode:
            self.mode_switches += 1
            logger.info(f"Mode switch for {self.sender_id}: {self.mode} → {new_mode}")
        self.mode = new_mode
        self.update_activity()
    
    def set_verification_stage(self, stage: str, **additional_data):
        """Set verification stage with additional data."""
        self.verification_stage = stage
        for key, value in additional_data.items():
            setattr(self, key, value)
        self.update_activity()
        
        logger.info(f"Verification stage updated for {self.sender_id}: {stage}")
    
    def is_expired(self, timeout_hours: int = 2) -> bool:
        """Check if session is expired."""
        return (datetime.now() - self.last_activity).seconds > (timeout_hours * 3600)
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get comprehensive session information."""
        return {
            "sender_id": self.sender_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "verification_stage": self.verification_stage,
            "mode": self.mode,
            "cnic": self.cnic,
            "name": self.name,
            "accounts": self.accounts,
            "selected_account": self.selected_account,
            "conversation_count": self.conversation_count,
            "mode_switches": self.mode_switches,
            "last_query_type": self.last_query_type,
            "session_duration_minutes": (datetime.now() - self.created_at).total_seconds() / 60
        }

# Enhanced session management
enhanced_sessions: Dict[str, EnhancedUserSession] = {}

def cleanup_old_processed_messages():
    """Clean up old processed messages to prevent memory leaks."""
    # Keep only last 2000 message IDs (increased for better duplicate detection)
    if len(processed_messages) > 2000:
        recent_messages = list(processed_messages)[-2000:]
        processed_messages.clear()
        processed_messages.update(recent_messages)
        logger.info("Cleaned up old processed messages")

def cleanup_old_sessions():
    """Enhanced cleanup of old user sessions."""
    current_time = datetime.now()
    sessions_to_remove = []
    
    # Clean up old user_sessions (legacy)
    for session_id, session_data in user_sessions.items():
        if session_data.get('timestamp') and (current_time - session_data['timestamp']).seconds > 7200:  # 2 hours
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del user_sessions[session_id]
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old legacy sessions")
    
    # Clean up enhanced sessions
    enhanced_sessions_to_remove = []
    for sender_id, session in enhanced_sessions.items():
        if session.is_expired(timeout_hours=2):
            enhanced_sessions_to_remove.append(sender_id)
    
    for sender_id in enhanced_sessions_to_remove:
        session_info = enhanced_sessions[sender_id].get_session_info()
        del enhanced_sessions[sender_id]
        logger.info(f"Cleaned up expired enhanced session: {sender_id} (duration: {session_info['session_duration_minutes']:.1f}min)")

def cleanup_old_authenticated_users():
    """Clean up old authenticated users to prevent memory leaks."""
    current_time = datetime.now()
    users_to_remove = []
    
    for sender_id, user_data in authenticated_users.items():
        if user_data.get('timestamp') and (current_time - user_data['timestamp']).seconds > 3600:  # 1 hour
            users_to_remove.append(sender_id)
    
    for sender_id in users_to_remove:
        del authenticated_users[sender_id]
        if users_to_remove:
            logger.info(f"Cleaned up {len(users_to_remove)} old authenticated users")

def periodic_cleanup():
    """Enhanced periodic cleanup with detailed logging."""
    before_messages = len(processed_messages)
    before_sessions = len(user_sessions)
    before_enhanced = len(enhanced_sessions)
    before_users = len(authenticated_users)
    
    cleanup_old_processed_messages()
    cleanup_old_sessions()
    cleanup_old_authenticated_users()
    
    logger.info(f"Periodic cleanup completed:")
    logger.info(f"  Messages: {before_messages} → {len(processed_messages)}")
    logger.info(f"  Legacy sessions: {before_sessions} → {len(user_sessions)}")
    logger.info(f"  Enhanced sessions: {before_enhanced} → {len(enhanced_sessions)}")
    logger.info(f"  Authenticated users: {before_users} → {len(authenticated_users)}")

# Enhanced helper functions
def get_enhanced_session(sender_id: str) -> EnhancedUserSession:
    """Get or create enhanced session for user."""
    if sender_id not in enhanced_sessions:
        enhanced_sessions[sender_id] = EnhancedUserSession(sender_id)
        logger.info(f"Created new enhanced session for {sender_id}")
    
    session = enhanced_sessions[sender_id]
    session.update_activity()
    return session

def get_user_verification_stage(sender_id: str) -> str:
    """Get current verification stage for user (enhanced)."""
    if sender_id in enhanced_sessions:
        return enhanced_sessions[sender_id].verification_stage
    
    # Fallback to legacy system
    if sender_id not in authenticated_users:
        return VERIFICATION_STAGES["NOT_VERIFIED"]
    return authenticated_users[sender_id].get("verification_stage", VERIFICATION_STAGES["NOT_VERIFIED"])

def set_user_verification_stage(sender_id: str, stage: str, **additional_data):
    """Set verification stage for user with enhanced session support."""
    # Update enhanced session
    session = get_enhanced_session(sender_id)
    session.set_verification_stage(stage, **additional_data)
    
    # Update legacy system for backward compatibility
    if sender_id not in authenticated_users:
        authenticated_users[sender_id] = {}
    
    authenticated_users[sender_id].update({
        "verification_stage": stage,
        "timestamp": datetime.now(),
        **additional_data
    })
    
    logger.info(f"Verification stage set: {sender_id} → {stage}")

def is_fully_authenticated(sender_id: str) -> bool:
    """Check if user is fully authenticated (enhanced)."""
    return get_user_verification_stage(sender_id) == VERIFICATION_STAGES["ACCOUNT_SELECTED"]

def get_user_account_info(sender_id: str) -> Dict[str, Any]:
    """Get user's selected account information (enhanced)."""
    if sender_id in enhanced_sessions:
        session = enhanced_sessions[sender_id]
        if session.verification_stage == VERIFICATION_STAGES["ACCOUNT_SELECTED"]:
            return {
                "cnic": session.cnic,
                "name": session.name,
                "account_number": session.selected_account,
                "verification_stage": session.verification_stage,
                "accounts": session.accounts,
                "mode": session.mode
            }
    
    # Fallback to legacy system
    if sender_id in authenticated_users and is_fully_authenticated(sender_id):
        user_data = authenticated_users[sender_id]
        return {
            "cnic": user_data.get("cnic"),
            "name": user_data.get("name"),
            "account_number": user_data.get("selected_account"),
            "verification_stage": user_data.get("verification_stage"),
            "accounts": user_data.get("accounts", []),
            "mode": user_data.get("mode", USER_MODES["ACCOUNT"])
        }
    
    return {}

def get_user_mode(sender_id: str) -> str:
    """Get current user mode (enhanced)."""
    if sender_id in enhanced_sessions:
        return enhanced_sessions[sender_id].mode
    
    # Fallback to legacy
    if sender_id in authenticated_users:
        return authenticated_users[sender_id].get("mode", USER_MODES["INITIAL"])
    
    return USER_MODES["INITIAL"]

def set_user_mode(sender_id: str, mode: str):
    """Set user mode (enhanced)."""
    session = get_enhanced_session(sender_id)
    session.switch_mode(mode)
    
    # Update legacy system for backward compatibility
    if sender_id in authenticated_users:
        authenticated_users[sender_id]["mode"] = mode
        authenticated_users[sender_id]["timestamp"] = datetime.now()

def update_user_query_context(sender_id: str, query_type: str, context_data: Dict[str, Any] = None):
    """Update user's query context for enhanced tracking."""
    session = get_enhanced_session(sender_id)
    session.last_query_type = query_type
    
    if context_data:
        session.context_data.update(context_data)

def get_user_session_stats(sender_id: str) -> Dict[str, Any]:
    """Get comprehensive user session statistics."""
    if sender_id in enhanced_sessions:
        return enhanced_sessions[sender_id].get_session_info()
    
    # Fallback for legacy users
    if sender_id in authenticated_users:
        user_data = authenticated_users[sender_id]
        return {
            "sender_id": sender_id,
            "verification_stage": user_data.get("verification_stage"),
            "mode": user_data.get("mode", USER_MODES["INITIAL"]),
            "cnic": user_data.get("cnic"),
            "name": user_data.get("name"),
            "legacy_session": True
        }
    
    return {"sender_id": sender_id, "no_session": True}

def clear_user_state(sender_id: str):
    """Clear all state data for a user (enhanced)."""
    # Clear enhanced session
    if sender_id in enhanced_sessions:
        session_info = enhanced_sessions[sender_id].get_session_info()
        del enhanced_sessions[sender_id]
        logger.info(f"Cleared enhanced session for {sender_id} (duration: {session_info['session_duration_minutes']:.1f}min)")
    
    # Clear legacy data
    if sender_id in authenticated_users:
        del authenticated_users[sender_id]
        logger.info(f"Cleared legacy state for {sender_id}")

def get_system_stats() -> Dict[str, Any]:
    """Get comprehensive system statistics."""
    active_sessions = len(enhanced_sessions)
    legacy_users = len(authenticated_users)
    processed_msgs = len(processed_messages)
    
    # Enhanced session breakdown
    verification_breakdown = {}
    mode_breakdown = {}
    
    for session in enhanced_sessions.values():
        stage = session.verification_stage
        mode = session.mode
        
        verification_breakdown[stage] = verification_breakdown.get(stage, 0) + 1
        mode_breakdown[mode] = mode_breakdown.get(mode, 0) + 1
    
    return {
        "enhanced_sessions": active_sessions,
        "legacy_authenticated_users": legacy_users,
        "processed_messages": processed_msgs,
        "verification_breakdown": verification_breakdown,
        "mode_breakdown": mode_breakdown,
        "total_active_users": active_sessions + legacy_users,
        "system_uptime": datetime.now().isoformat()
    }

# Export enhanced functions for backward compatibility
__all__ = [
    'authenticated_users', 'user_sessions', 'processed_messages',
    'VERIFICATION_STAGES', 'USER_MODES', 'EnhancedUserSession',
    'get_enhanced_session', 'get_user_verification_stage', 'set_user_verification_stage',
    'is_fully_authenticated', 'get_user_account_info', 'get_user_mode', 'set_user_mode',
    'update_user_query_context', 'get_user_session_stats', 'clear_user_state',
    'periodic_cleanup', 'get_system_stats'
]