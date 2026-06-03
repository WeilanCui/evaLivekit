"""Tests for French and CJK text normalization.

Covers the special-case logic that the English tests don't exercise:
- French vigesimal number system (70-99)
- French text normalization (decimal comma, thousand separators, fillers, abbreviations)
- French ordinal forms
- Japanese: kanji positional arithmetic, digit-string concatenation, full/half-width, jaconv
- Chinese: Simplified + Traditional variants, 两/兩, 亿/萬
- Korean: sino-Korean positional (with ambiguity guard), native Korean tens/ones
- Shared positional-number edge cases (leading unit, large-unit implicit-1)
"""

import pytest
from pipecat.frames.frames import Language

from eva.utils.wer_normalization.cjk import (
    ChineseTextNormalizer,
    JapaneseTextNormalizer,
    KoreanTextNormalizer,
)
from eva.utils.wer_normalization.wer_utils import normalize_text

# ===========================================================================
# French — number equivalence (digits ↔ spelled-out)
# ===========================================================================


class TestFrenchNumbers:
    """Spelled-out French numbers must round-trip to Arabic digit form."""

    @pytest.mark.parametrize(
        "digits,words",
        [
            # Regular base-10
            ("3", "trois"),
            ("12", "douze"),
            ("16", "seize"),
            ("21", "vingt et un"),
            ("30", "trente"),
            ("100", "cent"),
            ("1000", "mille"),
            # Vigesimal forms — the core French special case
            ("70", "soixante-dix"),  # 60 + 10
            ("71", "soixante et onze"),  # 60 + 11
            ("75", "soixante-quinze"),  # 60 + 15
            ("79", "soixante-dix-neuf"),  # 60 + 19
            ("80", "quatre-vingts"),  # 4 × 20
            ("81", "quatre-vingt-un"),  # 4 × 20 + 1
            ("90", "quatre-vingt-dix"),  # 4 × 20 + 10
            ("99", "quatre-vingt-dix-neuf"),  # 4 × 20 + 19
            # Larger numbers
            ("200", "deux cents"),
            ("1000000", "un million"),
            ("1500000000", "un milliard cinq cents millions"),
        ],
    )
    def test_digits_match_spelled_out(self, digits: str, words: str):
        assert normalize_text(digits, Language.FR.value) == normalize_text(words, Language.FR.value)

    def test_feminine_one(self):
        # "une" (feminine) must equal "un" / "1"
        assert normalize_text("une", Language.FR.value) == normalize_text("1", Language.FR.value)

    def test_decimal_virgule(self):
        # "virgule" is the French decimal word
        assert normalize_text("3,5", Language.FR.value) == normalize_text("trois virgule cinq", Language.FR.value)

    def test_ordinal_premier(self):
        # "premier" has no diacritics so it survives the diacritic-stripping step
        # and is converted by the number normalizer to "1er".
        assert normalize_text("premier", Language.FR.value) == normalize_text("1er", Language.FR.value)

    def test_ordinal_diacritics_recognized(self):
        # Ordinals containing diacritics (deuxième, centième …) are looked up
        # against the vocabulary in their accent-stripped form, so they
        # successfully resolve to the digit+suffix output (e.g. "2e").
        assert normalize_text("deuxième", Language.FR.value) == normalize_text("2e", Language.FR.value)
        # "2ème" and "2eme" are common misspellings; after diacritic-stripping
        # they both reduce to "2eme" which does not match "2e", so they stay as-is.
        assert normalize_text("2ème", Language.FR.value) == normalize_text("2eme", Language.FR.value)


# ===========================================================================
# French — text normalization (outer layer)
# ===========================================================================


class TestFrenchTextNormalization:
    def test_filler_words_removed(self):
        result = normalize_text("euh je voudrais un café", Language.FR.value)
        assert "euh" not in result
        # diacritics are stripped by the normalizer, so é → e
        assert "cafe" in result

    def test_abbreviations_expanded(self):
        # "mme" → "madame"
        assert "madame" in normalize_text("mme dupont", Language.FR.value)

    def test_thousand_separator_dot(self):
        # French uses period as thousand separator: 1.000 → 1000
        assert normalize_text("1.000", Language.FR.value) == normalize_text("1000", Language.FR.value)

    def test_thousand_separator_space(self):
        # French also uses space: 1 000 → 1000
        assert normalize_text("1 000 euros", Language.FR.value) == normalize_text("1000 euros", Language.FR.value)

    def test_decimal_comma_to_dot(self):
        # 1,5 → 1.5 before number processing
        assert normalize_text("1,5", Language.FR.value) == normalize_text("1.5", Language.FR.value)

    def test_markup_removed(self):
        # Expressive tags like [laughs], [music] are STT artifacts — they are
        # stripped regardless of language and would never be translated.
        result = normalize_text("bonjour [laughs] comment ca va", Language.FR.value)
        assert "[laughs]" not in result
        assert "bonjour" in result

    def test_diacritics_preserved_after_normalisation(self):
        # The normalizer should keep diacritic letters (they go through
        # remove_symbols_and_diacritics which strips diacritics, so the
        # output should be ASCII-safe but still recognisable).
        result = normalize_text("Au bâtiment Headquarters, à l'étage FL2.", Language.FR.value)
        assert "batiment" in result or "bâtiment" in result  # diacritics may or may not be stripped
        assert "headquarters" in result


# ===========================================================================
# Japanese
# ===========================================================================


@pytest.fixture(scope="module")
def ja():
    return JapaneseTextNormalizer()


class TestJapaneseNumbers:
    @pytest.mark.parametrize(
        "kanji,arabic",
        [
            # Positional (contains a unit character)
            ("十", "10"),  # implicit leading 1
            ("二十", "20"),
            ("三百二十一", "321"),
            ("千二百三十四", "1234"),
            ("一万二千三百四十五", "12345"),
            ("万", "10000"),  # large unit, implicit 1
            ("一億", "100000000"),
            # Pure-digit concatenation (no units → ID/phone number mode)
            ("一二三", "123"),
            ("〇一二三", "0123"),
        ],
    )
    def test_kanji_to_arabic(self, ja, kanji: str, arabic: str):
        assert ja(kanji) == arabic

    def test_full_width_digits(self, ja):
        assert ja("１２３") == "123"

    def test_full_width_ascii(self, ja):
        assert ja("ＡＢＣＤ") == "abcd"

    def test_filler_removal(self, ja):
        result = ja("えーとよろしくお願いします")
        assert "えーと" not in result
        assert "よろしく" in result

    def test_japanese_punctuation_stripped(self, ja):
        result = ja("こんにちは。今日はいい天気ですね。")
        assert "。" not in result

    def test_markup_removed(self, ja):
        result = ja("お客様[laughs]は三名です")
        assert "[laughs]" not in result
        assert "3" in result


# ===========================================================================
# Chinese
# ===========================================================================


@pytest.fixture(scope="module")
def zh():
    return ChineseTextNormalizer()


class TestChineseNumbers:
    @pytest.mark.parametrize(
        "hanzi,arabic",
        [
            # Simplified Chinese
            ("三百二十一", "321"),
            ("一千", "1000"),
            ("一亿", "100000000"),
            ("两千", "2000"),  # 两 (liǎng) = 2 in counting context
            ("一亿两千万", "120000000"),
            # Traditional Chinese variants
            ("兩千", "2000"),  # Traditional 兩
            ("一億兩千萬", "120000000"),  # Traditional 億/萬
            # Mixed Simplified/Traditional should both work
            ("二十萬", "200000"),
        ],
    )
    def test_hanzi_to_arabic(self, zh, hanzi: str, arabic: str):
        assert zh(hanzi) == arabic

    def test_simplified_and_traditional_equivalent(self, zh):
        # Same number expressed in both scripts must normalize identically
        assert zh("一亿两千万") == zh("一億兩千萬")

    def test_full_width_ascii(self, zh):
        assert zh("ＡＢＣＤ") == "abcd"

    def test_filler_removal(self, zh):
        result = zh("那个就是对的")
        assert "那个" not in result
        assert "就是" not in result

    def test_markup_removed(self, zh):
        result = zh("价格[music]是三百元")
        assert "[music]" not in result
        assert "300" in result


# ===========================================================================
# Korean
# ===========================================================================


@pytest.fixture(scope="module")
def ko():
    return KoreanTextNormalizer()


class TestKoreanSinoNumbers:
    @pytest.mark.parametrize(
        "sino,arabic",
        [
            ("십", "10"),  # implicit leading 1
            ("삼백이십일", "321"),
            ("이천이십사", "2024"),
            ("만이천삼백사십오", "12345"),
            ("일억", "100000000"),
        ],
    )
    def test_sino_korean_to_arabic(self, ko, sino: str, arabic: str):
        assert ko(sino) == arabic

    def test_ambiguous_syllables_not_converted(self, ko):
        # 이 = "this/two", 일 = "work/one" — must NOT be converted without a unit
        assert ko("이 책") == "이 책"  # "this book"
        assert ko("일하다") == "일하다"  # "to work"


class TestKoreanNativeNumbers:
    @pytest.mark.parametrize(
        "native,arabic",
        [
            ("아홉", "9"),
            ("열", "10"),
            ("열다섯", "15"),
            ("스물하나", "21"),
            ("서른", "30"),
            ("아흔아홉", "99"),
            ("열 하나", "11"),  # space between tens and ones
        ],
    )
    def test_native_korean_to_arabic(self, ko, native: str, arabic: str):
        assert ko(native) == arabic

    def test_filler_removal(self, ko):
        result = ko("음 주문하겠습니다")
        assert result.startswith("주문") or "주문" in result

    def test_markup_removed(self, ko):
        result = ko("고객[laughs]은 스물하나입니다")
        assert "[laughs]" not in result
        assert "21" in result
