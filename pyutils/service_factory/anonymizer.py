import datetime
import spacy
import json
import re


class Anonymizer(object):

    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        pass

    def __load_spacy_model(self, text_language:str, spacy_size_model:str):

        '''
        This private function loads a model for italian, english, french and german.
        :param text_language: Text language. Admitted values are italian, english, french and german only.
        :param spacy_size_model: Spacy size model. Admitted values are sm(small), md(medium), lg(large).
        :return: NLP model loaded using spacy if no exception raised (e.g. requested model is not downloaded before.)
        '''

        match text_language.lower():
            case 'italian':
                try:
                    nlp =  spacy.load("it_core_news_" + str(spacy_size_model).lower())
                    return nlp
                except:
                    raise Exception('Trying to load the italian model caused error. Is the model downloaded from Spacy ?' + \
                                    'Is model size correct ? Model size admitted are sm, md and lg only.')
            case 'english':
                try:
                    nlp = spacy.load("en_core_web_" + str(spacy_size_model).lower())
                    return nlp
                except:
                    raise Exception('Trying to load the english model caused error. Is the model downloaded from Spacy ?' + \
                                    'Is model size correct ? Model size admitted are sm, md and lg only.')
            case 'french':
                try:
                    nlp = spacy.load("fr_core_news_" + str(spacy_size_model).lower())
                    return nlp
                except:
                    raise Exception('Trying to load the french model caused error. Is the model downloaded from Spacy ?' + \
                                    'Is model size correct ? Model size admitted are sm, md and lg only.')
            case 'german':
                try:
                    nlp = spacy.load("de_core_news_" + str(spacy_size_model).lower())
                    return nlp
                except:
                    raise Exception(
                        'Trying to load the german model caused error. Is the model downloaded from Spacy ?' + \
                        'Is model size correct ? Model size admitted are sm, md and lg only.')
            case _:
                return False

    def __internal_text_cleaning(self, text_to_anonymize:str) -> str:

        # 1. Remove newlines
        # 2. Remove HTML tags
        # 3. Remove URLs
        # 4. Remove multiple spaces

        text_to_anonymize = text_to_anonymize.replace('\n', ' ')
        text_to_anonymize = re.sub(r'<.*?>', '', text_to_anonymize)
        text_to_anonymize = re.sub(r'http\S+|www\S+|https\S+', '', text_to_anonymize, flags=re.MULTILINE)
        text_to_anonymize = re.sub(r'\s+', ' ', text_to_anonymize).strip()

        return text_to_anonymize

    def __anonymize_text(self, model_loaded, text_to_anonymize:str):

        """
            Anonymizes named entities in the input text using a specified NLP model.

            This method processes the given text to identify and anonymize named entities (e.g., person names, organizations, locations, etc.) using a pre-loaded spaCy NLP model. Each identified entity is replaced with a generic label corresponding to its entity type concatenated with a unique identifier. The replacement aims to preserve the entity type information while removing the actual entity value to ensure anonymity. The method modifies the text by replacing each entity with its anonymized version and returns the modified text.

            Parameters:
            - model_loaded: A spaCy language model instance that has been loaded into memory. This model is used to identify named entities in the input text.
            - text_to_anonymize (str): The text to be anonymized. It is assumed that the text is a single string, possibly containing multiple sentences or paragraphs.

            Returns:
            - str: The anonymized text where each named entity is replaced with an anonymized label that reflects the entity's type and a unique identifier within that type in the format "<EntityType>_<Identifier>".

            Notes:
            - The function anonymizes entities by iterating through all entities identified by the spaCy model in the input text. It generates a unique label for each entity based on its type and a counter that increments for each occurrence of that type.
            - The method performs text replacement in a case-sensitive manner and does not account for variations in entity mentions (e.g., abbreviations, alternative spellings).
            - Line breaks in the input text are replaced with spaces to ensure consistent processing of multi-line texts.
            - This method is designed to be used as a private member of a class and thus is prefixed with double underscores.

            Example usage:
            ```python
            import spacy
            nlp = spacy.load("en_core_web_sm")
            anonymizer = SomeClass()
            anonymized_text = anonymizer.__anonymize_text(nlp, "Alice and Bob work at Google.")
            print(anonymized_text)
            # Output: "PERSON_1 and PERSON_2 work at ORG_1."
            ```
            """

        doc     = model_loaded(text_to_anonymize)
        labels  = {ent.label_ for ent in doc.ents}
        values  = {(str(ent), str(ent.label_)) for ent in doc.ents}
        val_lab = list()

        for label in labels:
            counter = 1
            for value in values:
                if value[1] == label:
                    val_lab.append( [value[0], str(value[1]) + '_' + str(counter)] )
                    counter += 1

        ###################################################################
        # Anonymize here below the text replacing VALUES with ENTITIES
        return_txt = text_to_anonymize.replace('\n', ' ')
        for val in val_lab:
            text_to_anonymize = return_txt.replace(val[0], val[1])
            return_txt        = text_to_anonymize

        return return_txt

    def anonymize_text(self,
                       text_to_anonymize:str  = '',
                       text_language:str      = '',
                       spacy_size_model:str   = '') -> str:

        if text_to_anonymize == '':
            raise Exception('Pass text to anonymize, no text passed till now.')

        if text_language == '':
            raise Exception('Pass the language of the text to anonymize, no language passed till now.' + \
                            'Admitted languages are italian, english, german, french only.')

        if spacy_size_model == '':
            raise Exception('Pass the language of the text to anonymize, no language passed till now.' + \
                            'Admitted spacy size models are are sm (small), md (medium), lg (large) only.')

        #######################################################################
        # 1. Load the SPACY downloaded model, first.
        model_loaded = self.__load_spacy_model(text_language, spacy_size_model)

        if not model_loaded:
            raise Exception('Attention! You tried to load a model that is not downloaded for the given language.' + \
                            'Pass one of the admitted languages models which are italian, english, german, french ONLY.')

        #######################################################################
        # 2. Clean the text before proceeding
        text_to_anonymize = self.__internal_text_cleaning(text_to_anonymize)

        #######################################################################
        # 3. Finally anonymize the text
        text_to_anonymize = self.__anonymize_text(text_to_anonymize = text_to_anonymize,
                                                  model_loaded      = model_loaded)

        return text_to_anonymize




# # Uncomment the following text to evaluate and test the function
#
# txt = '''
# On November 23, 2020, the Magistrate's Court of the District of Sonoma declared the bankruptcy of Nexus Ltd.
# On December 5, 2020, during the questioning of Attorney John Wayne it was revealed that the administrator had a severe damage and the company had a debt situation of about
# 900,000 USD. On December 14, 2024, the Bankruptcy Office of Sonoma requested the continuation of the
# bankruptcy liquidation in a summary manner. John Wayne, on January 29 2025, filed a claim for back his
# salaries. On February 1, 2025, Attorney Luke Skywalker, on behalf of Altura Inc., offered to withdraw
# the inventory of Nexus Ltd. for 500,000 USD. On September 30, the inventory of Nexus Ltd.'s assets
# was drafted. On January 10, 2026, the ranking of creditors was filed, with Altura Inc. registered for a
# claim secured by manual pledge of 5,000,000 USD for John Wayne. All well, what ends well at Street Los Angeles 1.
# '''
#
#
# analyzer = Anonymizer()
# analyzed = analyzer.anonymize_text(text_language='english',
#                                    text_to_anonymize=txt,
#                                    spacy_size_model='lg')

