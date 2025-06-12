import sqlite3
from dataclasses import asdict
from datetime import datetime

class SessionManager:
    def __init__(self, db_path: str = "juggling_sessions.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()

    def setup_database(self):
        """Create necessary tables if they don't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    throw_height_consistency REAL,
                    horizontal_drift REAL,
                    beat_timing_error REAL,
                    dwell_ratio REAL,
                    pattern_symmetry REAL
                )
            """)

    def save_session(self, metrics: AdvancedMetrics):
        """Save session metrics to database."""
        with self.conn:
            self.conn.execute("""
                INSERT INTO sessions 
                (timestamp, throw_height_consistency, horizontal_drift, 
                 beat_timing_error, dwell_ratio, pattern_symmetry)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), *astdict(metrics).values()))