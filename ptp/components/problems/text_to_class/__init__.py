from .dummy_language_identification import DummyLanguageIdentification
from .language_identification import LanguageIdentification
from .willy_language_identification import WiLYLanguageIdentification
from .wily_ngram_language_modeling import WiLYNGramLanguageModeling

__all__ = [
    'DummyLanguageIdentification',
    'LanguageIdentification',
    'WiLYLanguageIdentification',
    'WiLYNGramLanguageModeling'
    ]
