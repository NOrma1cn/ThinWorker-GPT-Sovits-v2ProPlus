# Minimal i18n stub: passthrough, no actual translation
class I18nAuto:
    def __init__(self, language: str = "Auto"):
        self.language = language

    def __call__(self, text: str) -> str:
        return text


def scan_language_list():
    return ["Auto", "zh", "en"]
