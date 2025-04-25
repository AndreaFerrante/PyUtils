from setuptools import setup, find_packages


setup(
    name                          = 'pyutils',
    version                       = '1.0.0',
    description                   = 'Utilities for everyday Python coding !',
    author                        = 'Andrea Ferrante',
    author_email                  = 'nonicknamethankyou@gmail.com',
    classifiers                   = ['Development Status :: Release',
                                     'Intended Audience :: Developers',
                                     'Topic :: Software Development :: Build Tools',
                                     'License :: OSI Approved :: MIT License',
                                     'Programming Language :: Python :: 3.6',
                                     'Programming Language :: Python :: 3.7',
                                     'Programming Language :: Python :: 3.8',
                                     'Programming Language :: Python :: 3.9',
                                     'Programming Language :: Python :: 3.10',
                                     'Programming Language :: Python :: 3.11',
                                     'Programming Language :: Python :: 3.12',
                                     'Programming Language :: Python :: 3.13',
                                     'Programming Language :: Python :: 3 :: Only'],
    keywords                      = 'datetime, setuptools',
    packages                      = find_packages(where='src'),
    python_requires               = '>=3.11',
    install_requires              = ['pandas',
                                     'numpy',
                                     'openai',
                                     'spacy',
                                     'tiktoken',
                                     'youtube-transcript-api'],
    py_modules                    = ['pyutils']
)
