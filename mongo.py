# Enhanced mongo.py - MongoDB connection and configuration for hybrid banking AI
import os
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from typing import Optional
import time

logger = logging.getLogger(__name__)

class EnhancedMongoConnection:
    """Enhanced MongoDB connection with better error handling and configuration."""
    
    def __init__(self, 
                 mongodb_uri: str = "mongodb://localhost:27017/",
                 db_name: str = "bank_database",
                 collection_name: str = "transactions"):
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        
        self.connect()
    
    def connect(self):
        """Establish connection to MongoDB with enhanced error handling."""
        try:
            logger.info(f"Connecting to MongoDB: {self.mongodb_uri}")
            
            # Create client with enhanced configuration
            self.client = MongoClient(
                self.mongodb_uri,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=10000,         # 10 second connection timeout
                socketTimeoutMS=20000,          # 20 second socket timeout
                maxPoolSize=50,                 # Maximum connection pool size
                retryWrites=True,               # Enable retryable writes
                retryReads=True                 # Enable retryable reads
            )
            
            # Test the connection
            self.client.admin.command('ping')
            
            # Set up database and collection
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            # Log connection info
            server_info = self.client.server_info()
            logger.info(f"✅ MongoDB connected successfully")
            logger.info(f"   Database: {self.db_name}")
            logger.info(f"   Collection: {self.collection_name}")
            logger.info(f"   MongoDB version: {server_info.get('version', 'unknown')}")
            
            # Get collection stats
            try:
                count = self.collection.count_documents({})
                logger.info(f"   Total documents: {count:,}")
                
                # Sample document structure
                sample_doc = self.collection.find_one()
                if sample_doc:
                    fields = list(sample_doc.keys())
                    logger.info(f"   Document fields: {fields}")
                
            except Exception as e:
                logger.warning(f"Could not get collection stats: {e}")
            
        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
        except ServerSelectionTimeoutError as e:
            logger.error(f"❌ MongoDB server selection timeout: {e}")
            logger.error("   Make sure MongoDB is running and accessible")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected MongoDB connection error: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test if MongoDB connection is still active."""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False
        return False
    
    def reconnect(self):
        """Reconnect to MongoDB if connection is lost."""
        try:
            if self.client:
                self.client.close()
        except:
            pass
        
        self.connect()
    
    def get_collection_stats(self) -> dict:
        """Get detailed collection statistics."""
        try:
            if not self.collection:
                return {"error": "No collection available"}
            
            # Basic stats
            total_docs = self.collection.count_documents({})
            
            # Get sample of unique field values for enhanced hybrid approach
            stats = {
                "total_documents": total_docs,
                "database": self.db_name,
                "collection": self.collection_name,
                "connection_uri": self.mongodb_uri.replace(self.mongodb_uri.split('@')[-1] if '@' in self.mongodb_uri else '', '***')  # Mask credentials
            }
            
            if total_docs > 0:
                # Sample document
                sample_doc = self.collection.find_one()
                if sample_doc:
                    stats["sample_fields"] = list(sample_doc.keys())
                
                # Get unique account numbers
                unique_accounts = len(list(self.collection.distinct("account_number")))
                stats["unique_accounts"] = unique_accounts
                
                # Get unique CNICs
                unique_cnics = len(list(self.collection.distinct("cnic")))
                stats["unique_users"] = unique_cnics
                
                # Currency breakdown
                currency_breakdown = {}
                for currency in self.collection.distinct("transaction_currency"):
                    count = self.collection.count_documents({"transaction_currency": currency})
                    currency_breakdown[currency] = count
                stats["currency_breakdown"] = currency_breakdown
                
                # Transaction type breakdown
                type_breakdown = {}
                for tx_type in self.collection.distinct("type"):
                    count = self.collection.count_documents({"type": tx_type})
                    type_breakdown[tx_type] = count
                stats["type_breakdown"] = type_breakdown
                
                # Date range
                try:
                    oldest = self.collection.find().sort("date", 1).limit(1)[0]["date"]
                    newest = self.collection.find().sort("date", -1).limit(1)[0]["date"]
                    stats["date_range"] = {
                        "oldest": oldest.isoformat() if oldest else None,
                        "newest": newest.isoformat() if newest else None
                    }
                except:
                    stats["date_range"] = {"error": "Could not determine date range"}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close MongoDB connection."""
        try:
            if self.client:
                self.client.close()
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

# Create enhanced global connection instance
try:
    # Try to get configuration from environment variables
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    DB_NAME = os.getenv("DB_NAME", "bank_database")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "transactions")
    
    # Initialize enhanced connection
    mongo_connection = EnhancedMongoConnection(
        mongodb_uri=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME
    )
    
    # Export the collection for direct use (backward compatibility)
    transactions = mongo_connection.collection
    db = mongo_connection.db
    client = mongo_connection.client
    
    logger.info("Enhanced MongoDB connection initialized and exported successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize enhanced MongoDB connection: {e}")
    logger.error("This will cause API endpoints to fail. Please check your MongoDB setup.")
    
    # Create dummy objects to prevent import errors
    transactions = None
    db = None
    client = None
    mongo_connection = None

def get_mongo_health() -> dict:
    """Get MongoDB health status for health checks."""
    if not mongo_connection:
        return {
            "status": "unhealthy",
            "error": "MongoDB connection not initialized",
            "connected": False
        }
    
    try:
        is_connected = mongo_connection.test_connection()
        stats = mongo_connection.get_collection_stats() if is_connected else {}
        
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "connected": is_connected,
            "stats": stats,
            "database": mongo_connection.db_name,
            "collection": mongo_connection.collection_name
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "connected": False
        }

def reconnect_if_needed():
    """Reconnect to MongoDB if connection is lost."""
    if mongo_connection and not mongo_connection.test_connection():
        logger.warning("MongoDB connection lost, attempting to reconnect...")
        try:
            mongo_connection.reconnect()
            logger.info("MongoDB reconnection successful")
        except Exception as e:
            logger.error(f"MongoDB reconnection failed: {e}")

# Export all necessary objects and functions
__all__ = [
    'transactions', 'db', 'client', 'mongo_connection',
    'get_mongo_health', 'reconnect_if_needed', 'EnhancedMongoConnection'
]