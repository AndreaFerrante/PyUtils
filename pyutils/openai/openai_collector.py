import time
import base64
import requests
import argparse
import tiktoken
import pandas as pd
from openai import OpenAI
from datetime import datetime


class OpenAICollector:

    def __init__(self, api_key:str, content:str="", max_retries:int=5, timeout:float=125.0, model:str="", embedder:str=""):

        self.api_key     = api_key
        self.timeout     = timeout
        self.max_retries = max_retries
        self.model       = self.get_model(model)
        self.embedder    = self.get_embedder(embedder)
        self.content     = self.get_content(content)
        self.client      = self.__create_openai_client()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"api_key='***', "  # Masked for security
            f"content={self.content!r}, "
            f"max_retries={self.max_retries!r}, "
            f"timeout={self.timeout!r}, "
            f"model={self.model!r}, "
            f"embedder={self.embedder!r})"
        )

    @staticmethod
    def encode_image(image_path:str):

        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as ex:
            raise Exception(f"While encoding the image, this error occured: {ex}")

    @staticmethod
    def convert_unix_datetime(timestamp:int, format:str='%Y-%m-%d %H:%M:%S') -> str:

        try:
            return datetime.fromtimestamp(timestamp).strftime(format)
        except Exception as ex:
            raise Exception(f"While converting unix datetime, this exception occured: {ex}")

    @staticmethod
    def get_content(content:str) -> str:

        if len(content):
            return content

        return '''You are an AI assistant and a professional Python coder with extensive expertise in machine learning. 
        You work with precision and always take the time to carefully craft the best possible answer.'''

    @staticmethod
    def get_model(model:str) -> str:

        if len(model):
            return model

        return "gpt-4.1"

    @staticmethod
    def get_embedder(embedder: str) -> str:

        if len(embedder):
            return embedder

        return "text-embedding-3-large"

    def __create_openai_client(self) -> OpenAI:

        try:

            client = OpenAI(api_key     = self.api_key,
                            max_retries = self.max_retries,
                            timeout     = self.timeout)

            return client

        except Exception as ex:
            raise Exception(f"While creating OpenAI client this exception occured: {ex}")

    def get_openai_models_dataframe(self, timeout:float=30.0) -> pd.DataFrame:

        try:

            openai_url = "https://api.openai.com/v1/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            model_name              = list()
            model_object            = list()
            model_creation_datetime = list()
            model_owned_by          = list()

            response      = requests.get(openai_url, headers=headers, timeout=timeout)
            response_json = response.json()
            response_data = response_json['data']

            for response_data_el in response_data:
                model_name.append(response_data_el['id'])
                model_object.append(response_data_el['object'])
                model_creation_datetime.append(response_data_el['created'])
                model_owned_by.append(response_data_el['owned_by'])

            models = pd.DataFrame({
                'model_name':               model_name,
                'model_object':             model_object,
                'model_creation_datetime': [self.convert_unix_datetime(x) for x in model_creation_datetime],
                'model_owned_by':          model_owned_by
            }).sort_values(['model_name', 'model_creation_datetime'], ascending=[True, False])

            return models

        except requests.exceptions.RequestException as ex:
            raise Exception(f"While requesting all OpenAI models, this error occured: {ex}")

    def get_embeddings(self, text_to_embed:list=None, embedder:str="", return_full_object:bool=False, timer:bool=False):

        try:

            if len(embedder):
                embedder = self.embedder

            start_time = time.time()
            embeddings = self.client.embeddings.create(
                            model = embedder,
                            input = text_to_embed)
            embedding_time = time.time() - start_time

            if timer:
                print(f"Embedding took: {round(embedding_time, 4)} seconds.")

            if return_full_object:
                return embeddings

            return embeddings.data

        except Exception as ex:
            raise Exception(f"While embedding, this error occured: {ex}")

    def get_answer_given_query(self, query:str="", model:str="", timer:bool=False):

        ''' All openai models overview: https://platform.openai.com/docs/models/overview  '''

        try:

            if model == "":
                model = self.model

            start_time = time.time()
            response = self.client.chat.completions.create(
                    model    = model,
                    messages = [
                        {"role": "system",
                         "content": self.content},
                        {"role": "user",
                         "content": str(query)}
                    ]
                )
            answer_time = time.time() - start_time

            if timer:
                print(f"Answer took: {round(answer_time, 3)} seconds.")

            return response.choices[0].message.content

        except Exception as ex:
            raise Exception(f"While posting a query to {model}, this error occured: {ex}")

    def get_reasoned_answer_given_query(self, query:str="", model:str="o1"):

        ''' All openai models overview: https://platform.openai.com/docs/models/overview  '''

        try:

            if model == "":
                model = self.model

            response = self.client.chat.completions.create(
                    model    = model,
                    messages = [
                        {"role": "system",
                         "content": self.content},
                        {"role": "user",
                         "content": query}
                    ]
                )

        except Exception as ex:
            raise Exception(f"While posting a query to {model}, this error occured: {ex}")

        return response.choices[0].message.content

    def get_tokens_in_string(self, text_to_tokenize:str, encoding_model:str="") -> int:

        try:

            if encoding_model == "":
                encoding_model = self.model

            encoding   = tiktoken.encoding_for_model(encoding_model)
            num_tokens = len(encoding.encode(text_to_tokenize))

        except Exception:
            print(f"Fallback encoding model: cl100k_base. Model {encoding_model} has no encoding method, yet.")
            encoding   = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text_to_tokenize))

        return num_tokens


def main():
    
    parser=argparse.ArgumentParser(
    description='Manages OpenAI API class interface to submit prompts',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog= """
            Examples:

            """
    )
    
    parser.add_argument(
        '--query',
        '-q',
        required=True,
        help='Prompt to submit to OpenAI'
    )
    
    parser.add_argument(
        '--api_key',
        '-ak',
        required=True,
        help='Submit API Key for OpenAI'
    )
    
    parser.add_argument(
        '--content',
        '-c',
        required=False,
        help='OpenAI system content'
    )
    
    parser.add_argument(
        '--model',
        '-m',
        required=False,
        help='OpenAI model to be used (default GPT 4o)'
    )
    
    args = parser.parse_args()
    
    openai_collector = OpenAICollector(api_key = args.api_key,
                                       conten  = args.content)
    
if __name__ == '__main__':
    main()
