from langdetect import detect, LangDetectException

def is_french(text: str) -> bool:
    """
    Retourne True si le texte est détecté comme français.
    Retourne False en cas d'erreur ou de texte trop court.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        return False
    try:
        return detect(text) == "fr"
    except LangDetectException:
        return False