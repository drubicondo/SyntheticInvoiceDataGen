"""FLOPayments ML - Synthetic Invoice and Payment Data Generator"""

__version__ = "1.0.0"
__author__ = "IASON ITALIA S.R.L"

from .generators.synthetic_data_generator import SyntheticDataGenerator
from .generators.ai_text_generator import AITextGenerator
from .core.data_models import Fattura, Transazione
from .core.data_types import MatchType, QualityLevel, TimingPattern, AmountPattern, GroundTruth
from .config.settings import DEFAULT_CONFIG

__all__ = [
    'SyntheticDataGenerator',
    'AITextGenerator', 
    'Fattura',
    'Transazione',
    'MatchType',
    'QualityLevel', 
    'TimingPattern',
    'AmountPattern',
    'GroundTruth',
    'DEFAULT_CONFIG'
]