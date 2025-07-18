from enum import Enum
from dataclasses import dataclass

class MatchType(Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    RELATED = "related"
    UNRELATED = "unrelated"

class QualityLevel(Enum):
    PERFECT = "perfect"
    FUZZY = "fuzzy"
    NOISY = "noisy"

class TimingPattern(Enum):
    STANDARD = "standard"  # 0-90 days
    DELAYED = "delayed"    # >90 days
    EARLY = "early"       # before invoice
    SAME_DAY = "same_day"  # same day

class AmountPattern(Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    EXCESS = "excess"
    DISCOUNT = "discount"
    PENALTY = "penalty"

@dataclass
class GroundTruth:
    fattura_id: str
    pagamento_id: str
    match_type: str
    confidence: float
    amount_covered: float
    notes: str

class EnvironmentConfigError(Exception):
    """Raised when environment variables cannot be loaded or are missing"""
    pass