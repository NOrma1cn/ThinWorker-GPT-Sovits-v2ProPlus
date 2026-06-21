from thin_tts.text import symbols2

_symbol_to_id_v2 = {s: i for i, s in enumerate(symbols2.symbols)}


def cleaned_text_to_sequence(cleaned_text, version="v2"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text."""
    phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]
    return phones
