"""Lightweight text normalization for CER comparison.

No dependency on qwen_tts or other heavy NLP packages.
"""

import re
from typing import Callable, Optional

import cn2an


class TextNormalizer:
    """Text normalizer for CER computation.

    Supports Chinese + alphanumeric text normalization.
    Optionally accepts an external preprocessor (e.g., cn2an for number conversion).
    """

    def __init__(self, external_preprocessor: Optional[Callable[[str, str], str]] = None):
        """
        Args:
            external_preprocessor: Optional callable with signature
                (text: str, lang: str) -> str. Applied before regex cleanup.
        """
        self._preprocessor = external_preprocessor

    def normalize(self, text: str, language: str = "zh") -> str:
        """Normalize text for CER comparison.

        Steps:
        1. Lowercase
        2. Optional external preprocessing
        3. Remove instruction control tags: [...] and <...>
        4. Strip all characters except CJK, ASCII letters, and digits
        5. Convert Arabic numerals to Chinese numerals (e.g., "25" \u2192 "\u4e8c\u5341\u4e94")

        Args:
            text: Input text string.
            language: Language hint for external preprocessor.

        Returns:
            Normalized text containing only CJK/alpha/digit characters.
        """
        text = text.lower()
        if self._preprocessor:
            try:
                text = self._preprocessor(text, language)
            except Exception:
                pass
        # Remove instruction control tags like [energy_high], <end_ins>, etc.
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'<.*?>', '', text)
        # Keep only Chinese characters, letters, and digits
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        # Convert Arabic numerals to Chinese numerals for consistent comparison
        text = cn2an.transform(text, 'an2cn')
        return text
