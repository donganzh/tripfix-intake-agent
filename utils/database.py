import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

class IntakeDatabase:
    def __init__(self, db_path: str = "data/database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create intake sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intake_sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'in_progress',
                flight_data TEXT,
                jurisdiction TEXT,
                jurisdiction_confidence REAL,
                eligibility_result TEXT,
                eligibility_confidence REAL,
                compensation_amount REAL,
                legal_citations TEXT,
                handoff_reason TEXT,
                completed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create conversation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_type TEXT,
                content TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES intake_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_id: str) -> bool:
        """Create a new intake session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO intake_sessions (id) VALUES (?)
            ''', (session_id,))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session with new data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build dynamic update query
            fields = []
            values = []
            for key, value in kwargs.items():
                if key in ['flight_data', 'legal_citations', 'handoff_reason']:
                    fields.append(f"{key} = ?")
                    values.append(json.dumps(value) if isinstance(value, (dict, list)) else value)
                else:
                    fields.append(f"{key} = ?")
                    values.append(value)
            
            fields.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(session_id)
            
            query = f"UPDATE intake_sessions SET {', '.join(fields)} WHERE id = ?"
            cursor.execute(query, values)
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating session: {e}")
            return False
    
    def add_message(self, session_id: str, message_type: str, content: str, metadata: Dict = None):
        """Add a message to conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversation_history (session_id, message_type, content, metadata)
            VALUES (?, ?, ?, ?)
        ''', (session_id, message_type, content, json.dumps(metadata) if metadata else None))
        conn.commit()
        conn.close()
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM intake_sessions WHERE id = ?', (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def get_conversation_history(self, session_id: str) -> list:
        """Get conversation history for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM conversation_history 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        ''', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]