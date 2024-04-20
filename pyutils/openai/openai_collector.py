import openai
import tiktoken
from openai import OpenAI

class OpenAICollector(object):

    def __init__(self, api_key, content:str=''):
        super().__init__()
        self.api_key = api_key
        self.content = content # This defines the personality of the bot.
        self.client  = OpenAI(api_key=self.api_key)

    def __get_moderation_given_prompt(self):
        pass

    def get_embeddings(self, text_to_embed:list=None, model:str='text-embedding-ada-002', return_full_object:bool=False):

        """
        Here - general embedding model openai api https://platform.openai.com/docs/guides/embeddings

        Get embeddings for a list of text using the specified model.

        Parameters:
            text_to_embed (list): List of strings to embed.
            model (str): Name of the embedding model to use. Default is 'text-embedding-ada-002'.
            return_full_object (bool): Whether to return the full embedding object or just the data. Default is False.

        Returns:
            Union[dict, object]: Embedding data or full embedding object based on return_full_object parameter.

        Raises:
            Exception: If text_to_embed is None or if there's an error during embedding creation.

        Example:
            embedding = get_embedding(text_to_embed=["example text"], model="text-embedding-ada-002", return_full_object=False)
        """

        if text_to_embed is None:
            raise Exception('Pass text to embed, first.')

        if model not in ['text-embedding-ada-002','text-embedding-3-large','text-embedding-3-small']:
            raise Exception("Attention, the model can be only 'text-embedding-3-small','text-embedding-3-large', 'text-embedding-ada-002'.")

        try:
            embeddings = self.client.embeddings.create(
                            model = model,
                            input = text_to_embed)
        except Exception as ex:
            raise ex

        if return_full_object:
            return embeddings

        return embeddings.data

    def get_answer_give_prompt(self, prompt:str='', model:str='gpt-3.5-turbo'):

        '''
        Here - general openai model overview: https://platform.openai.com/docs/models/overview

        Get an answer based on a given prompt using the specified model.

        Parameters:
            prompt (str): The prompt to generate the answer.
            model (str): Name of the model to use. Default is 'gpt-3.5-turbo'.

        Returns:
            str: Generated answer based on the prompt.

        Raises:
            Exception: If prompt is an empty string or if there's an error during response generation.

        Example:
            answer = get_answer_give_prompt(prompt="What is the meaning of life?", model="gpt-3.5-turbo")
        '''

        if prompt == '':
            raise Exception('Pass a prompt to have an answer for.')

        try:
            if self.content == '':
                self.content = 'You are an accurate assistant'

            response = self.client.chat.completions.create(
                    model    = model,
                    messages = [
                        {"role": "system",
                         "content": self.content},
                        {"role": "user",
                         "content": prompt}
                    ]
                )
        except Exception as ex:
            raise ex

        return response.choices[0].message.content

    def get_tokens_in_string(self, text_to_measure:str = "", model_encoding:str='gpt-3.5-turbo'):

        """
            Calculates the number of tokens in a given text based on the specified model's tokenization standards.

            This function uses the 'tiktoken' library to tokenize the input text according to the encoding standard
            of a specified model. It is useful for understanding how many tokens a particular text would generate
            when processed by models like GPT-3.5 Turbo.

            Parameters:
                text_to_measure (str): The text for which tokens need to be counted. Defaults to an empty string.
                model_encoding (str): The name of the model based on whose encoding the text will be tokenized.
                                      Defaults to 'gpt-3.5-turbo'.

            Returns:
                int: The number of tokens that the input text generates according to the specified model's encoding.

            Raises:
                Exception: If no text is provided to the function (i.e., text_to_measure is an empty string).

            Example:
                >>> self.get_tokens_in_string("Hello world!", "gpt-3.5-turbo")
                3  # Assuming "Hello world!" translates to 3 tokens in the specified model encoding.

            Note:
                This function depends on the 'tiktoken' library, which must be correctly installed and configured
                to use the encoding_for_model function and to support the specified model encodings.
            """

        if text_to_measure == "":
            raise Exception("Pass a text for token measurament to function 'get_tokens_in_string'.")

        if model_encoding not in ['gpt-3.5-turbo', 'gpt-4']:
            raise Exception("model_encoding param can be 'gpt-4' or 'gpt-3.5-turbo' only")

        encoding   = tiktoken.encoding_for_model(model_encoding)
        num_tokens = len(encoding.encode(text_to_measure))

        return num_tokens

    def get_images(self):
        pass




