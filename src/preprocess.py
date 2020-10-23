import nltk
import string
import json

data_source = './resources/books.txt'

print('Downloading NLTK Corpora and Models')
nltk.download(info_or_id='all', quiet=True)

regular_docs = []
token_docs = []
with open(data_source, 'r') as file:
    for line in file:
        regular_docs.append(line.replace('\n', ''))
        tokens = list(map((lambda x: x.lower()), filter((lambda x: x not in string.punctuation), nltk.word_tokenize(line))))
        token_docs.append(tokens)

with open('./resources/preprocess', 'w') as file:
    file.write(json.dumps(list(regular_docs)) + '\n')
    file.write(json.dumps(token_docs))
