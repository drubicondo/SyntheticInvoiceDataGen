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
    STANDARD = "standard"
    DELAYED = "delayed"
    EARLY = "early"
    SAME_DAY = "same_day"

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