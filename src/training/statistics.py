from dataclasses import dataclass, field
from datetime import datetime
import sqlite3
from typing import List, Dict, Optional
import numpy as np
from ..analysis.metrics import AdvancedMetrics
from .drill_generator import DrillExercise

@dataclass
class DrillStats:
    drill_name: str
    focus_metric: str
    start_time: datetime
    duration: float
    success_rate: float
    initial_value: float
    final_value: float
    improvement: float

class StatisticsManager:
    def __init__(self, db_path: str = "juggling_stats.db"):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database for statistics tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drill_history (
                    id INTEGER PRIMARY KEY,
                    drill_name TEXT,
                    focus_metric TEXT,
                    start_time TIMESTAMP,
                    duration REAL,
                    success_rate REAL,
                    initial_value REAL,
                    final_value REAL,
                    improvement REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    throw_height_consistency REAL,
                    horizontal_drift REAL,
                    beat_timing_error REAL,
                    dwell_ratio REAL,
                    pattern_symmetry REAL
                )
            """)
    
    def save_drill_stats(self, stats: DrillStats):
        """Save completed drill statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO drill_history (
                    drill_name, focus_metric, start_time, duration,
                    success_rate, initial_value, final_value, improvement
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats.drill_name, stats.focus_metric, stats.start_time,
                stats.duration, stats.success_rate, stats.initial_value,
                stats.final_value, stats.improvement
            ))
    
    def save_session_metrics(self, metrics: AdvancedMetrics):
        """Save session metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO session_metrics (
                    timestamp, throw_height_consistency, horizontal_drift,
                    beat_timing_error, dwell_ratio, pattern_symmetry
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(), metrics.throw_height_consistency,
                metrics.horizontal_drift, metrics.beat_timing_error,
                metrics.dwell_ratio, metrics.pattern_symmetry
            ))
    
    def get_metric_history(self, metric_name: str, days: int = 7) -> Dict[datetime, float]:
        """Get historical values for a specific metric."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT timestamp, {metric_name} FROM session_metrics
                WHERE timestamp >= date('now', '-{days} days')
                ORDER BY timestamp ASC
            """)
            return {row[0]: row[1] for row in cursor.fetchall()}
    
    def get_improvement_rate(self, metric_name: str) -> Optional[float]:
        """Calculate improvement rate for a metric."""
        history = self.get_metric_history(metric_name)
        if len(history) < 2:
            return None
        values = list(history.values())
        return (values[-1] - values[0]) / values[0]

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

@dataclass
class MetricHistory:
    values: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    max_points: int = 1000

    def add_value(self, value: float) -> None:
        try:
            if not isinstance(value, (int, float)):
                raise ValueError("Invalid metric value type")
                
            self.values.append(float(value))
            self.timestamps.append(datetime.now())
            
            # Maintain fixed buffer size
            if len(self.values) > self.max_points:
                self.values = self.values[-self.max_points:]
                self.timestamps = self.timestamps[-self.max_points:]
                
        except Exception as e:
            print(f"Error adding metric value: {e}")

class StatsTracker:
    def __init__(self):
        self.metrics: Dict[str, MetricHistory] = {
            'height_consistency': MetricHistory([], []),
            'horizontal_drift': MetricHistory([], []),
            'beat_timing': MetricHistory([], []),
            'dwell_ratio': MetricHistory([], []),
            'pattern_symmetry': MetricHistory([], [])
        }
    
    def update_metrics(self, metrics_dict: Dict[str, float]):
        """Update all metrics with new values."""
        for metric_name, value in metrics_dict.items():
            if metric_name in self.metrics:
                self.metrics[metric_name].add_value(value)