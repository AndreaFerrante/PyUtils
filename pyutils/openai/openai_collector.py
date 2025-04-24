import requests
import tiktoken
import numpy as np
import pandas as pd
from openai import OpenAI
from datetime import datetime


class OpenAICollector:

    def __init__(self, api_key:str, content:str="", max_retries:int=5, timeout:float=30.0, model:str="", embedder:str=""):

        self.api_key     = api_key
        self.timeout     = timeout
        self.max_retries = max_retries
        self.model       = self.__get_model(model)
        self.embedder    = self.__get_embedder(embedder)
        self.content     = self.__get_content(content)
        self.client      = self.__create_openai_client()

    @staticmethod
    def __convert_unix_datetime(timestamp:int, format:str='%Y-%m-%d %H:%M:%S') -> str:

        try:
            return datetime.fromtimestamp(timestamp).strftime(format)
        except Exception as ex:
            raise Exception(f"While converting unix datetime, this exception occured: {ex}")

    @staticmethod
    def __get_content(content:str) -> str:

        if len(content):
            return content

        return '''You are an AI assistant and a professional Python coder with extensive expertise in machine learning. 
        You work with precision and always take the time to carefully craft the best possible answer.'''

    @staticmethod
    def __get_model(model:str) -> str:

        if len(model):
            return model

        return "gpt-4.1"

    @staticmethod
    def __get_embedder(embedder: str) -> str:

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
                'model_creation_datetime': [self.__convert_unix_datetime(x) for x in model_creation_datetime],
                'model_owned_by':          model_owned_by
            }).sort_values(['model_name', 'model_creation_datetime'], ascending=[True, False])

            return models

        except requests.exceptions.RequestException as ex:
            raise Exception(f"While requesting all OpenAI models, this error occured: {ex}")

    def get_embeddings(self, text_to_embed:list=None, embedder:str="", return_full_object:bool=False):

        try:

            if len(embedder):
                embedder = self.embedder

            embeddings = self.client.embeddings.create(
                            model = embedder,
                            input = text_to_embed)

            if return_full_object:
                return embeddings

            return embeddings.data

        except Exception as ex:
            raise Exception(f"While embedding, this error occured: {ex}")

    def get_answer_given_query(self, query:str="", model:str=""):

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
