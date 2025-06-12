from dataclasses import dataclass
from typing import Tuple
import yaml

@dataclass
class TrackingConfig:
    min_ball_radius: int
    max_ball_radius: int
    color_lower: Tuple[int, int, int]
    color_upper: Tuple[int, int, int]
    max_disappeared: int

@dataclass
class Config:
    tracking: TrackingConfig

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        tracking = TrackingConfig(**data['tracking'])
        return cls(tracking=tracking)