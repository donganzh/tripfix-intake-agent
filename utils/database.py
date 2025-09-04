import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
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
                handoff_priority TEXT,
                risk_level TEXT,
                risk_assessment TEXT,
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
        
        # Create supporting files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS supporting_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                filename TEXT,
                file_type TEXT,
                file_size INTEGER,
                file_path TEXT,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                extracted_text TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES intake_sessions (id)
            )
        ''')
        
        # Create intake progress table to track what information has been collected
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intake_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                flight_number_collected BOOLEAN DEFAULT FALSE,
                flight_date_collected BOOLEAN DEFAULT FALSE,
                airline_collected BOOLEAN DEFAULT FALSE,
                origin_collected BOOLEAN DEFAULT FALSE,
                destination_collected BOOLEAN DEFAULT FALSE,
                connecting_airports_collected BOOLEAN DEFAULT FALSE,
                delay_length_collected BOOLEAN DEFAULT FALSE,
                delay_reason_collected BOOLEAN DEFAULT FALSE,
                supporting_files_offered BOOLEAN DEFAULT FALSE,
                intake_complete BOOLEAN DEFAULT FALSE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES intake_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Migrate existing databases to add new columns
        self._migrate_schema()
    
    def _migrate_schema(self):
        """Migrate existing database schema to add new columns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if new columns exist and add them if they don't
        cursor.execute("PRAGMA table_info(intake_sessions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        new_columns = [
            ('handoff_priority', 'TEXT'),
            ('risk_level', 'TEXT'),
            ('risk_assessment', 'TEXT')
        ]
        
        for column_name, column_type in new_columns:
            if column_name not in columns:
                try:
                    cursor.execute(f'ALTER TABLE intake_sessions ADD COLUMN {column_name} {column_type}')
                    print(f"Added column {column_name} to intake_sessions table")
                except sqlite3.OperationalError as e:
                    print(f"Column {column_name} might already exist: {e}")
        
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
                if key in ['flight_data', 'legal_citations', 'handoff_reason', 'risk_assessment']:
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
    
    def add_supporting_file(self, session_id: str, filename: str, file_type: str, 
                           file_size: int, file_path: str, extracted_text: str = None, 
                           metadata: Dict = None) -> bool:
        """Add a supporting file to the session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO supporting_files 
                (session_id, filename, file_type, file_size, file_path, extracted_text, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, filename, file_type, file_size, file_path, 
                  extracted_text, json.dumps(metadata) if metadata else None))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding supporting file: {e}")
            return False
    
    def get_supporting_files(self, session_id: str) -> list:
        """Get all supporting files for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM supporting_files 
            WHERE session_id = ? 
            ORDER BY upload_timestamp ASC
        ''', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    
    def update_intake_progress(self, session_id: str, **kwargs) -> bool:
        """Update intake progress for a session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if progress record exists
            cursor.execute('SELECT id FROM intake_progress WHERE session_id = ?', (session_id,))
            if not cursor.fetchone():
                # Create new progress record
                cursor.execute('''
                    INSERT INTO intake_progress (session_id) VALUES (?)
                ''', (session_id,))
            
            # Update progress fields
            fields = []
            values = []
            for key, value in kwargs.items():
                if key in ['flight_number_collected', 'flight_date_collected', 'airline_collected',
                          'origin_collected', 'destination_collected', 'connecting_airports_collected',
                          'delay_length_collected', 'delay_reason_collected', 'supporting_files_offered',
                          'intake_complete']:
                    fields.append(f"{key} = ?")
                    values.append(value)
            
            fields.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(session_id)
            
            query = f"UPDATE intake_progress SET {', '.join(fields)} WHERE session_id = ?"
            cursor.execute(query, values)
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating intake progress: {e}")
            return False
    
    def get_intake_progress(self, session_id: str) -> Optional[Dict]:
        """Get intake progress for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM intake_progress WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    
    def is_intake_complete(self, session_id: str) -> bool:
        """Check if intake is complete for a session"""
        progress = self.get_intake_progress(session_id)
        if not progress:
            return False
        
        required_fields = [
            'flight_number_collected', 'flight_date_collected', 'airline_collected',
            'origin_collected', 'destination_collected', 'delay_length_collected',
            'delay_reason_collected'
        ]
        
        return all(progress.get(field, False) for field in required_fields)
    
    def get_completed_sessions(self) -> List[Dict[str, Any]]:
        """Get all completed intake sessions with detailed information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                id,
                created_at,
                updated_at,
                status,
                flight_data,
                jurisdiction,
                jurisdiction_confidence,
                eligibility_result,
                eligibility_confidence,
                compensation_amount,
                legal_citations,
                handoff_reason,
                handoff_priority,
                risk_level,
                risk_assessment,
                completed
            FROM intake_sessions 
            WHERE completed = 1 OR status IN ('eligibility_assessed', 'human_review_required', 'completed')
            ORDER BY created_at DESC
        ''')
        
        columns = [description[0] for description in cursor.description]
        sessions = []
        
        for row in cursor.fetchall():
            session = dict(zip(columns, row))
            sessions.append(session)
        
        conn.close()
        return sessions