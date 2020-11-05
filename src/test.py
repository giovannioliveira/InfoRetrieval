from main import handleQuery
import datetime
import nltk

Stemmer = nltk.stem.snowball.EnglishStemmer().stem

misspells_source = '../resources/misspells.txt'
misspells = {}

with open(misspells_source, 'r') as file:
    for line in file:
        aux = line.split('->')
        correction = (aux[1].replace('\n','')).split(', ')
        if len(correction) == 1:
            misspells[aux[0]] = correction[0]

print(datetime.datetime.now())
wrongs = []
for (wrong, right) in misspells.items():
    result = handleQuery(wrong)
    if Stemmer(result[0]) != Stemmer(right):
        wrongs.append((wrong, right, result))

print(datetime.datetime.now())

print(str(len(wrongs))+' out of '+str(len(misspells))+' wrong corrections')
for el in wrongs:
    print(el)
