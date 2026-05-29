"""
Tests pour le module clean/
"""

from sherlock.clean.language import is_french
from sherlock.clean.regex_rules import (
    clean_text,
    count_words_excluding_token,
    remove_hashtag_symbol,
    remove_urls,
    replace_mentions,
)

# ── Tests regex_rules ─────────────────────────────────────────────────────────


def test_remove_url():
    assert "bonjour" in remove_urls("bonjour https://example.com fin")
    assert "https" not in remove_urls("texte https://example.com")


def test_replace_mention():
    result = replace_mentions("bonjour @JeanDupont comment vas-tu")
    assert "@JeanDupont" not in result
    assert "__MENTION__" in result


def test_remove_hashtag_symbol():
    result = remove_hashtag_symbol("vive #LaFrance et #Liberté")
    assert "#" not in result
    assert "LaFrance" in result
    assert "Liberté" in result


def test_clean_text_full():
    raw = "Bonjour @user ! #Politique https://example.com 🎉"
    result = clean_text(raw)
    assert "https" not in result
    assert "@user" not in result
    assert "#" not in result
    assert result == result.lower()


def test_clean_text_not_string():
    assert clean_text(None) == ""
    assert clean_text(42) == ""


def test_count_words_excluding_token():
    text = "le président __mention__ a parlé __mention__ hier"
    assert count_words_excluding_token(text) == 5


def test_count_words_empty():
    assert count_words_excluding_token("") == 0
    assert count_words_excluding_token(None) == 0


# ── Tests language ────────────────────────────────────────────────────────────


def test_is_french_true():
    assert is_french("Le gouvernement a annoncé de nouvelles mesures économiques") is True


def test_is_french_false():
    assert is_french("The government announced new economic measures") is False


def test_is_french_too_short():
    assert is_french("ok") is False


def test_is_french_none():
    assert is_french(None) is False
