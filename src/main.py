import itertools
import numpy
import nltk
import string
import json
from spellchecker import SpellChecker
from rank_bm25 import BM25Okapi

#Stem = nltk.stem.snowball.EnglishStemmer().stem

regular_docs = None
dictionary = None
token_docs = None

preprocess_file = '../resources/preprocess'
alt_threshold = 16

with open(preprocess_file, 'r') as file:
    regular_docs = json.loads(file.readline())
    token_docs = json.loads(file.readline())

spell = SpellChecker(language=None)
spell.word_frequency.load_text_file(preprocess_file)

bm25 = BM25Okapi(token_docs)


def get_scores(query):
    result = bm25.get_scores(query)
    for i in range(len(result)):
        if result[i] != 0:
            if len(set(query).difference(set(token_docs[i]))) != 0:
                result[i] = 0
    return result


def get_best_suggestion(suggestions):
    max_score = 0
    max_suggestion = None
    for s in suggestions:
        max_s = get_scores(s).sum()
        if max_s > max_score:
            max_score = max_s
            max_suggestion = s
    return max_suggestion


def get_least_worst_suggestion(suggestions):
    max_score = 0
    max_suggestion = None
    for s in suggestions:
        max_s = 0
        for w in s:
            max_s += bm25.get_scores([w]).sum()
        if max_s > max_score:
            max_score = max_s
            max_suggestion = s
    return max_suggestion

def handleQuery(query):
    token_query = list(set(map((lambda x: x.lower()),
                           filter((lambda x: x not in string.punctuation), nltk.word_tokenize(query)))))
    scores = get_scores(token_query)

    if numpy.count_nonzero(scores) == 0:
        word_sets = []
        if len(spell.unknown(token_query)) > 0:
            word_sets = list([x] for x in spell.known(token_query))
            word_sets += list(list(spell.candidates(w)) for w in spell.unknown(token_query))
            suggestions = list(itertools.product(*word_sets))
            best_suggestion = get_best_suggestion(suggestions)
            if not best_suggestion:
                best_suggestion = get_least_worst_suggestion(suggestions)
            if not best_suggestion:
                best_suggestion = token_query
            token_query = best_suggestion
            scores = get_scores(token_query)
        if numpy.count_nonzero(scores) == 0:
            word_sets = []
            for w in token_query:
                alt = spell.known(spell.edit_distance_1(w))
                word_sets.append([w] if len(alt) > alt_threshold else list(alt))
            suggestions = list(itertools.product(*word_sets))
            best_suggestion = get_best_suggestion(suggestions)
            if not best_suggestion:
                best_suggestion = get_least_worst_suggestion(suggestions)
            if not best_suggestion:
                best_suggestion = token_query
            token_query = best_suggestion
        print('Changing null query to: ' + ' '.join(token_query))
    return token_query


if __name__ == "__main__":
    while True:
        query = input()
        token_query = handleQuery(query)
        scores = get_scores(token_query)

        if numpy.count_nonzero(scores) == 0:
            print('Definitive null query')
        else:
            results = []
            for (i, value) in enumerate(scores):
                if value != 0:
                    results.append((regular_docs[i], value))
            results.sort(reverse=True, key=(lambda x: x[1]))
            for item in results:
                print(item)
