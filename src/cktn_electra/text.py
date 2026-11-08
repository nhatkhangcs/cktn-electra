"""Token-surface helpers used by the replacement sampler.

Each vocabulary entry is assigned a coarse *script class* and a *noise* flag.
The sampler uses them to reject replacements that a discriminator could detect
from surface form alone (for example a Khmer-script piece inside a
Latin-script Cham or Tay-Nung sentence).
"""

from __future__ import annotations

import unicodedata

SCRIPT_SHARED = 0  # no alphabetic characters (digits, symbols): compatible with every script
SCRIPT_LATIN = 1
SCRIPT_KHMER = 2
SCRIPT_OTHER = 3

_KHMER_BLOCK = ("ក", "៿")  # Unicode Khmer block


def token_surface(token: str) -> str:
    """Strip SentencePiece/WordPiece markers before inspecting a token."""
    return token.replace("▁", "").replace("##", "").strip()


def is_noise_surface(surface: str) -> bool:
    """True for punctuation-only, digit-dominant or formatting-only surfaces."""
    chars = [ch for ch in surface if not ch.isspace()]
    if not chars:
        return True

    alpha_count = sum(ch.isalpha() for ch in chars)
    digit_count = sum(ch.isdigit() for ch in chars)
    symbol_count = sum(unicodedata.category(ch)[0] in {"P", "S", "C"} for ch in chars)
    return alpha_count == 0 or digit_count > alpha_count or symbol_count == len(chars)


def script_profile(surface: str) -> int:
    """Return the coarse script class of a token surface."""
    has_khmer = any(_KHMER_BLOCK[0] <= ch <= _KHMER_BLOCK[1] for ch in surface)
    has_latin = any(ch.isalpha() and "LATIN" in unicodedata.name(ch, "") for ch in surface)
    has_alpha = any(ch.isalpha() for ch in surface)

    if has_khmer and not has_latin:
        return SCRIPT_KHMER
    if has_latin and not has_khmer:
        return SCRIPT_LATIN
    if not has_alpha:
        return SCRIPT_SHARED
    return SCRIPT_OTHER
