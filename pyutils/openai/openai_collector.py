import time
import base64
import argparse

import requests
import tiktoken
import pandas as pd
from openai import OpenAI
from datetime import datetime


class OpenAICollector:
    """OpenAI API client with chat, embeddings, vision, and token-counting utilities."""

    DEFAULT_MODEL    = "gpt-4o-mini"
    DEFAULT_EMBEDDER = "text-embedding-3-large"
    DEFAULT_CONTENT  = (
        "You are an AI assistant and a professional Python coder with extensive "
        "expertise in machine learning. You work with precision and always take "
        "the time to carefully craft the best possible answer."
    )

    def __init__(
        self,
        api_key:     str,
        content:     str   = "",
        max_retries: int   = 5,
        timeout:     float = 125.0,
        model:       str   = "",
        embedder:    str   = "",
    ) -> None:
        self.api_key     = api_key
        self.timeout     = timeout
        self.max_retries = max_retries
        self.model       = model    or self.DEFAULT_MODEL
        self.embedder    = embedder or self.DEFAULT_EMBEDDER
        self.content     = content  or self.DEFAULT_CONTENT
        self.client      = OpenAI(
            api_key     = self.api_key,
            max_retries = self.max_retries,
            timeout     = self.timeout,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"api_key='***', "
            f"content={self.content!r}, "
            f"max_retries={self.max_retries!r}, "
            f"timeout={self.timeout!r}, "
            f"model={self.model!r}, "
            f"embedder={self.embedder!r})"
        )

    @staticmethod
    def encode_image(image_path: str) -> str:
        """Return the base64-encoded contents of an image file."""
        try:
            with open(image_path, "rb") as fh:
                return base64.b64encode(fh.read()).decode("utf-8")
        except OSError as ex:
            raise OSError(f"Could not encode image '{image_path}': {ex}") from ex

    @staticmethod
    def convert_unix_datetime(timestamp: int, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Convert a Unix timestamp integer to a formatted datetime string."""
        return datetime.fromtimestamp(timestamp).strftime(format)

    def get_openai_models_dataframe(self, timeout: float = 30.0) -> pd.DataFrame:
        """Return available OpenAI models as a DataFrame sorted by creation date."""
        url = "https://api.openai.com/v1/models"
        try:
            response = requests.get(
                url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as ex:
            raise RuntimeError(f"Failed to fetch OpenAI models: {ex}") from ex

        data = response.json()["data"]
        df = pd.DataFrame({
            "model_name":              [m["id"]         for m in data],
            "model_object":            [m["object"]     for m in data],
            "model_creation_datetime": [self.convert_unix_datetime(m["created"]) for m in data],
            "model_owned_by":          [m["owned_by"]   for m in data],
        })
        return df.sort_values("model_creation_datetime", ascending=False).reset_index(drop=True)

    def get_embeddings(
        self,
        text_to_embed: list,
        embedder:      str  = "",
        return_full_object: bool = False,
        timer:         bool = False,
    ):
        """Return embeddings for text_to_embed using the specified (or default) embedder."""
        if not embedder:
            embedder = self.embedder
        try:
            start = time.time()
            embeddings = self.client.embeddings.create(model=embedder, input=text_to_embed)
            if timer:
                print(f"Embedding took: {round(time.time() - start, 4)} seconds.")
            return embeddings if return_full_object else embeddings.data
        except Exception as ex:
            raise RuntimeError(f"Embedding failed: {ex}") from ex

    def get_answer_given_query(self, query: str = "", model: str = "", timer: bool = False) -> str:
        """Submit a single-turn query and return the assistant's response text."""
        if not model:
            model = self.model
        try:
            start = time.time()
            response = self.client.chat.completions.create(
                model    = model,
                messages = [
                    {"role": "system", "content": self.content},
                    {"role": "user",   "content": str(query)},
                ],
            )
            if timer:
                print(f"Answer took: {round(time.time() - start, 3)} seconds.")
            return response.choices[0].message.content
        except Exception as ex:
            raise RuntimeError(f"Chat completion failed for model {model!r}: {ex}") from ex

    def get_reasoned_answer_given_query(self, query: str = "", model: str = "o1") -> str:
        """Submit a query to a reasoning model and return the response text."""
        if not model:
            model = self.model
        try:
            response = self.client.chat.completions.create(
                model    = model,
                messages = [
                    {"role": "system", "content": self.content},
                    {"role": "user",   "content": query},
                ],
            )
            return response.choices[0].message.content
        except Exception as ex:
            raise RuntimeError(f"Reasoned completion failed for model {model!r}: {ex}") from ex

    def get_tokens_in_string(self, text_to_tokenize: str, encoding_model: str = "") -> int:
        """Return the token count of text_to_tokenize using tiktoken."""
        if not encoding_model:
            encoding_model = self.model
        try:
            encoding = tiktoken.encoding_for_model(encoding_model)
        except KeyError:
            print(f"Fallback encoding model: cl100k_base. Model {encoding_model!r} has no encoding yet.")
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text_to_tokenize))


def main() -> None:
    parser = argparse.ArgumentParser(
        description     = "OpenAICollector — submit prompts to the OpenAI API",
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query",   "-q",  required=True,  help="Prompt to submit")
    parser.add_argument("--api_key", "-ak", required=True,  help="OpenAI API key")
    parser.add_argument("--content", "-c",  default="",     help="System prompt override")
    parser.add_argument("--model",   "-m",  default="",     help="Model name")

    args = parser.parse_args()
    collector = OpenAICollector(api_key=args.api_key, content=args.content, model=args.model)
    print(collector.get_answer_given_query(args.query))


if __name__ == "__main__":
    main()
