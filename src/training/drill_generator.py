from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import time
from datetime import datetime
from ..analysis.metrics import AdvancedMetrics, ThrowMetrics
from ..analysis.pattern_recognition import PatternInfo, JugglingPattern
from .statistics import StatisticsManager, DrillStats

@dataclass
class DrillConfig:
    duration_seconds: int = 30
    improvement_threshold: float = 0.2
    drill_types: List[str] = None

@dataclass
class DrillExercise:
    name: str
    description: str
    focus_metric: str
    target_value: float
    current_value: float
    pattern: JugglingPattern
    instructions: List[str]
    success_count: int = 0
    attempt_count: int = 0

class DrillGenerator:
    def __init__(self, config: DrillConfig = None):
        self.config = config or DrillConfig()
        self.drill_history: List[DrillExercise] = []
        self.current_drill: Optional[DrillExercise] = None
        self.drill_start_time = 0
        self.success_threshold = 0.8  # 80% success rate required
        self.stats_manager = StatisticsManager()
        self.initial_metrics = None
    
    def generate_drill(self, metrics: AdvancedMetrics, pattern_info: PatternInfo) -> DrillExercise:
        """Generate drill based on metrics and history."""
        if not self.initial_metrics:
            self.initial_metrics = metrics
            
        # Get improvement rates for all metrics
        improvement_rates = {
            'height_consistency': self.stats_manager.get_improvement_rate('throw_height_consistency'),
            'horizontal_drift': self.stats_manager.get_improvement_rate('horizontal_drift'),
            'timing': self.stats_manager.get_improvement_rate('beat_timing_error'),
            'dwell_ratio': self.stats_manager.get_improvement_rate('dwell_ratio'),
            'symmetry': self.stats_manager.get_improvement_rate('pattern_symmetry')
        }
        
        # Prioritize metrics with lowest improvement rate
        metric_scores = {}
        for metric, current_value in self._get_metric_values(metrics).items():
            rate = improvement_rates.get(metric, 0) or 0
            metric_scores[metric] = current_value * (1 - rate)
        
        sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
        focus_metric, _ = sorted_metrics[0]
        
        return self._create_drill_for_metric(focus_metric, metric_scores[focus_metric], pattern_info.pattern_type)
    
    def _create_drill_for_metric(self, 
                               metric: str, 
                               current_value: float,
                               pattern: JugglingPattern) -> DrillExercise:
        """Create specific drill exercise based on the metric to improve."""
        drills = {
            'height_consistency': self._height_consistency_drill,
            'horizontal_drift': self._horizontal_drift_drill,
            'timing': self._timing_drill,
            'dwell_ratio': self._dwell_ratio_drill,
            'symmetry': self._symmetry_drill
        }
        
        return drills[metric](current_value, pattern)
    
    def _height_consistency_drill(self, current_value: float, 
                                pattern: JugglingPattern) -> DrillExercise:
        """Create drill for improving throw height consistency."""
        return DrillExercise(
            name="Height Calibration",
            description="Practice consistent throw heights with visual feedback",
            focus_metric="height_consistency",
            target_value=current_value * 0.8,  # Aim for 20% improvement
            current_value=current_value,
            pattern=pattern,
            instructions=[
                "1. Focus on one ball at a time",
                "2. Use marked height zones on wall/screen as reference",
                "3. Aim for consistent peak heights",
                "4. Start slow, increase speed as consistency improves"
            ]
        )

    def _horizontal_drift_drill(self, current_value: float, 
                              pattern: JugglingPattern) -> DrillExercise:
        """Create drill for reducing horizontal drift."""
        return DrillExercise(
            name="Precision Landing",
            description="Improve throw accuracy with target spots",
            focus_metric="horizontal_drift",
            target_value=current_value * 0.8,
            current_value=current_value,
            pattern=pattern,
            instructions=[
                "1. Place markers on floor for target landing spots",
                "2. Start with larger targets, shrink as accuracy improves",
                "3. Focus on throwing straight up and down",
                "4. Keep elbows close to body"
            ]
        )

    def evaluate_progress(self, metrics: AdvancedMetrics) -> bool:
        """Check if current drill has achieved its improvement target."""
        if not self.drill_history:
            return False
            
        current_drill = self.drill_history[-1]
        current_value = getattr(metrics, current_drill.focus_metric)
        return (current_drill.current_value - current_value) / current_drill.current_value >= self.config.improvement_threshold
    
    def update_progress(self, metrics: AdvancedMetrics) -> float:
        """Update drill progress and return completion percentage."""
        if not self.current_drill:
            return 0.0

        # Calculate progress based on time
        elapsed = time.time() - self.drill_start_time
        progress = min(1.0, elapsed / self.config.duration_seconds)

        # Check if current throw meets success criteria
        current_value = getattr(metrics, self.current_drill.focus_metric)
        is_successful = current_value <= self.current_drill.target_value
        
        if is_successful:
            self.current_drill.success_count += 1
        self.current_drill.attempt_count += 1

        return progress
    
    def complete_drill(self, metrics: AdvancedMetrics):
        """Record drill completion statistics."""
        if not self.current_drill or not self.initial_metrics:
            return
            
        current_value = getattr(metrics, self.current_drill.focus_metric)
        initial_value = getattr(self.initial_metrics, self.current_drill.focus_metric)
        
        stats = DrillStats(
            drill_name=self.current_drill.name,
            focus_metric=self.current_drill.focus_metric,
            start_time=datetime.fromtimestamp(self.drill_start_time),
            duration=time.time() - self.drill_start_time,
            success_rate=self.current_drill.success_count / max(1, self.current_drill.attempt_count),
            initial_value=initial_value,
            final_value=current_value,
            improvement=(initial_value - current_value) / initial_value
        )
        
        self.stats_manager.save_drill_stats(stats)
        self.stats_manager.save_session_metrics(metrics)