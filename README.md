This package has different datetime and file system utilities that are handy for everyday coding life :tada: <br>
Open command prompt / terminal from the folder and install the PyUtils package with the command:

	pip install .

To remove the package open command in folder and install the PyUtils package with the command:

	pip uninstall pyutils

One class that is handy it the class named ```Anonymizer```: this class leverages (```spacy```)[https://spacy.io/] to mask names, dates, organisation (etc) in any given text.
Try to use it as follows:

```
txt = '''
On November 23, 2020, the Magistrate's Court of the District of Sonoma declared the bankruptcy of Nexus Ltd.
On December 5, 2020, during the questioning of Attorney John Wayne it was revealed that the administrator
had a severe damage and the company had a debt situation of about 900,000 USD.
On December 14, 2024, the Bankruptcy Office of Sonoma requested the continuation of the
bankruptcy liquidation. John Wayne, on January 29 2025, filed a claim for back his
salaries. On February 1, 2025, Attorney Luke Skywalker, on behalf of Altura Inc., offered to withdraw
the inventory of Nexus Ltd. for 500,000 USD. On September 30, the inventory of Nexus Ltd.'s assets
was drafted. On January 10, 2026, the ranking of creditors was filed, with Altura Inc. registered for a
claim secured by manual pledge of 5,000,000 USD for John Wayne. All well, what ends well at Street Los Angeles 1.
'''


analyzer = Anonymizer()
analyzed = analyzer.anonymize_text(text_language='english',
                                   text_to_anonymize=txt,
                                   spacy_size_model='lg')
```

Have fun and happy coding 🥳
