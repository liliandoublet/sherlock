import re
import emoji

URL_RE      = re.compile(r"https?://\S+")
MENTION_RE  = re.compile(r"@\w{1,15}")
HASHTAG_RE  = re.compile(r"#(\w+)")
QUOTE_RE    = re.compile(r"[\"«»\u201c\u201d\u2018\u2019']")
SPACES_RE   = re.compile(r"\s+")

def remove_emojis(text: str) -> str:
    return emoji.replace_emoji(text, "")

def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)

def replace_mentions(text: str, token: str = "__MENTION__") -> str:
    """Remplace @handle par un token neutre."""
    return MENTION_RE.sub(token, text)

def remove_hashtag_symbol(text: str) -> str:
    """#politique -> politique (garde le mot, supprime le #)."""
    return HASHTAG_RE.sub(r"\1", text)

def remove_quotes(text: str) -> str:
    return QUOTE_RE.sub(" ", text)

def normalize_spaces(text: str) -> str:
    return SPACES_RE.sub(" ", text).strip()


def clean_text(text: str, mention_token: str = "__MENTION__") -> str:
    """
    Nettoyage complet d'un texte (tweet ou communiqué).
    Ordre des opérations :
      1. Emojis supprimés
      2. URLs supprimées
      3. Mentions remplacées par mention_token
      4. Hashtags : # supprimé, mot conservé
      5. Guillemets supprimés
      6. Espaces normalisés + lowercase
    """
    if not isinstance(text, str):
        return ""
    text = remove_emojis(text)
    text = remove_urls(text)
    text = replace_mentions(text, mention_token)
    text = remove_hashtag_symbol(text)
    text = remove_quotes(text)
    text = normalize_spaces(text)
    return text.lower()


def count_words_excluding_token(text: str, token: str = "__MENTION__") -> int:
    """
    Compte les mots d'un texte en ignorant le token de mention.
    Utilisé pour filtrer les tweets trop courts (< 5 mots réels).
    """
    if not isinstance(text, str):
        return 0
    return sum(1 for w in text.split() if w != token.lower())