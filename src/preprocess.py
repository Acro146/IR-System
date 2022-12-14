from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Set
from io import StringIO


class StopWords:

    def __init__(self, unwanted_stop_words: Set):
        self.stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                           "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                           'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                           'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                           'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                           'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                           'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                           'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                           'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                           'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                           'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                           'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
                           'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                           "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
                           "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                           "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                           'wouldn', "wouldn't"}

        self.stop_words.difference_update(unwanted_stop_words)

    def __contains__(self, word: str) -> bool:
        return word in self.stop_words


class Scanner(metaclass=ABCMeta):
    text: StringIO

    def __call__(self, text):
        self.text = text
        return self

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class TokensScanner(Scanner):
    def __iter__(self):
        token = ""
        for char in iter(partial(self.text.read, 1), ""):
            if char.isalpha() or char == "'":
                token += char
            elif token:
                yield token
                token = ""
        if token:
            yield token


@dataclass
class Normalizer:
    def __call__(self, token: str) -> str:
        terms = token.lower()
        return terms


@dataclass
class Tokenizer:
    stop_list: StopWords
    normalizer: Normalizer
    scanner: Scanner
    text: StringIO = field(default_factory=lambda: StringIO())

    def __call__(self, text):
        self.text = text
        return self

    def __iter__(self):
        for token in self.scanner(self.text):
            term = self.normalizer(token)
            if term in self.stop_list:
                continue
            yield term
