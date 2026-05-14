import re
import spacy


class Anonymizer:
    """NLP-based text anonymizer using spaCy named-entity recognition."""

    _LANGUAGE_MODELS: dict[str, str] = {
        "italian": "it_core_news_",
        "english": "en_core_web_",
        "french":  "fr_core_news_",
        "german":  "de_core_news_",
    }

    def _load_spacy_model(self, text_language: str, spacy_size_model: str):
        """Load the spaCy model for the given language and size (sm/md/lg)."""
        prefix = self._LANGUAGE_MODELS.get(text_language.lower())
        if prefix is None:
            raise ValueError(
                f"Unsupported language {text_language!r}. "
                f"Supported: {', '.join(self._LANGUAGE_MODELS)}"
            )
        model_name = prefix + spacy_size_model.lower()
        try:
            return spacy.load(model_name)
        except OSError as ex:
            raise RuntimeError(
                f"Could not load spaCy model {model_name!r}. "
                "Is the model downloaded? Run: python -m spacy download <model>"
            ) from ex

    def _clean_text(self, text: str) -> str:
        """Strip newlines, HTML tags, URLs, and extra whitespace."""
        text = text.replace("\n", " ")
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.MULTILINE)
        return re.sub(r"\s+", " ", text).strip()

    def _anonymize_text(self, model, text: str) -> str:
        """Replace named entities with type-keyed labels (e.g. PERSON_1)."""
        doc = model(text)
        replacements: list[tuple[str, str]] = []
        counters: dict[str, int] = {}
        for ent in doc.ents:
            counters[ent.label_] = counters.get(ent.label_, 0) + 1
            replacements.append((ent.text, f"{ent.label_}_{counters[ent.label_]}"))

        for original, label in replacements:
            text = text.replace(original, label)
        return text

    def anonymize_text(
        self,
        text_to_anonymize: str,
        text_language: str,
        spacy_size_model: str,
    ) -> str:
        """Anonymize named entities in text using the specified spaCy model.

        Args:
            text_to_anonymize: Raw input text.
            text_language:     One of italian, english, french, german.
            spacy_size_model:  spaCy model size — sm, md, or lg.

        Returns:
            Anonymized text with entities replaced by TYPE_N labels.
        """
        if not text_to_anonymize:
            raise ValueError("text_to_anonymize must not be empty.")
        if not text_language:
            raise ValueError("text_language must not be empty.")
        if not spacy_size_model:
            raise ValueError("spacy_size_model must not be empty (sm / md / lg).")

        model = self._load_spacy_model(text_language, spacy_size_model)
        text_to_anonymize = self._clean_text(text_to_anonymize)
        return self._anonymize_text(model, text_to_anonymize)
