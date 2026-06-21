from thin_tts.text import chinese2
from thin_tts.text import symbols2

symbols = symbols2.symbols

special = [
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
]


def clean_text(text, language="zh", version="v2"):
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)

    norm_text = chinese2.text_normalize(text)
    phones, word2ph = chinese2.g2p(norm_text)
    assert len(phones) == sum(word2ph)
    assert len(norm_text) == len(word2ph)
    phones = ["UNK" if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    text = text.replace(special_s, ",")
    norm_text = chinese2.text_normalize(text)
    phones, word2ph = chinese2.g2p(norm_text)
    new_ph = []
    for ph in phones:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, word2ph, norm_text
