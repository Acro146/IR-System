import pickle
from collections import OrderedDict
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from .document_collection import Document, DocumentCollection
from .preprocess import Tokenizer


@dataclass
class Token:
    term: str
    position: int
    document: Document


@dataclass
class Term:
    tf: int = field(default=0)
    df: int = field(default=0)
    idf: float = field(default=0.0)
    tf_idf: float = field(default=0.0)


@dataclass(frozen=True)
class Candidate:
    position: int
    document: Document


@dataclass(order=True)
class Posting:
    document: Document
    frequency: int = field(default=0, compare=False)
    positions: list[int] = field(default_factory=lambda: [])

    def update(self, position):
        self.frequency += 1
        self.positions.append(position)

    def get_candidate(self, other_posting, k):
        index = 0
        other_position_index = 0
        positions = self.positions
        other_posting_positions = other_posting.positions
        other_posting_positions_length = len(other_posting_positions)
        positions_length = len(positions)
        result = set()

        while index != positions_length and other_position_index != other_posting_positions_length:
            position = positions[index]
            other_position = other_posting_positions[other_position_index]
            if other_position - position == k:
                result.add(Candidate(position, self.document))
                index += 1
                other_position_index += 1
            elif other_position - position > k:
                index += 1
            else:
                other_position_index += 1
        return result

    def print_posting(self):
        print("-" * 50)
        print(f"Document: '{self.document.document_path.name}', Frequency: {self.frequency} \nPositions: {self.positions}")


@dataclass
class PostingsList:
    term: str
    frequency: int = field(default=0, compare=False)
    postings: list[Posting] = field(default_factory=lambda: [])

    def update(self, token: Token) -> None:
        if not self.postings:
            self.postings.append(Posting(token.document, 1, [token.position]))
            self.frequency += 1
            return

        last_posting = self.postings[-1]
        if last_posting.document.document_id == token.document.document_id:
            last_posting.update(token.position)
        else:
            self.postings.append(Posting(token.document, 1, [token.position]))
            self.frequency += 1

    def __len__(self):
        return len(self.postings)

    def get_intersect(self, other_postings_list, k: int) -> set:
        posting_index = 0
        other_posting_index = 0
        postings = self.postings
        other_postings = other_postings_list.postings
        other_postings_length = len(other_postings)
        postings_length = len(postings)
        result = set()

        while posting_index != postings_length and other_posting_index != other_postings_length:
            posting = postings[posting_index]
            other_posting = other_postings[other_posting_index]

            if posting.document.document_id == other_posting.document.document_id:
                # two postings are from the same document then we can get the candidates from the postings
                # [term1:doc1:[1], term2:doc1:[2]]
                # search for term[k] in /k words from term1
                result = result.union(posting.get_candidate(other_posting, k))
                posting_index += 1
                other_posting_index += 1
            elif posting.document.document_id > other_posting.document.document_id:
                other_posting_index += 1
            else:
                posting_index += 1
        return result

    def get_postings(self) -> set:
        result = set()
        for posting in self.postings:
            for position in posting.positions:
                result.add(Candidate(position, posting.document))
        return result

    def print_postings(self):
        print("\n" + "*" * 25 + "Term" + "*" * 25)
        print(f"Term: '{self.term}', Document Frequency: {self.frequency}")
        for posting in self.postings:
            posting.print_posting()


@dataclass
class PositionalIndex:
    document_collection: DocumentCollection
    number_of_documents: int
    tokenizer: Tokenizer
    dictionary: OrderedDict[str, PostingsList]
    terms: OrderedDict[str, Term]

    def __init__(self, tokenizer, document_collection):
        self.tokenizer = tokenizer
        self.document_collection = document_collection
        self.get_dictionary()
        self.sort_postings()
        self.dictionary = OrderedDict(sorted(self.dictionary.items()))

    def sort_postings(self):
        if self.document_collection.directory is not None:
            for postings_list in self.dictionary.values():
                postings_list.postings.sort()

    def get_dictionary(self):
        self.number_of_documents = 0
        self.dictionary = OrderedDict()
        for document in self.document_collection:
            self.number_of_documents += 1
            with open(document.document_path) as file:
                for i, term in enumerate(self.tokenizer(file)):
                    postings_list = self.dictionary.get(term)
                    if not postings_list:
                        postings_list = PostingsList(term)
                    self.dictionary[term] = postings_list
                    postings_list.update(Token(term, i, document))

    def phrase_query(self, phrase: StringIO) -> set:
        terms = self.tokenizer(phrase)
        postings_lists = []
        result = set()

        for term in terms:
            if self.dictionary.get(term) is not None:
                postings_lists.append(self.dictionary.get(term))
        postings_lists_length = len(postings_lists)

        if postings_lists_length == 0:
            return result

        result = postings_lists[0].get_postings()

        if postings_lists_length == 1:
            return result
        # k words
        for k, postings in enumerate(postings_lists[1:], 1):
            result = result.intersection(postings_lists[0].get_intersect(postings, k))
        return result

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    def save(self, file_name):
        with open(Path(self.document_collection.directory) / file_name, "wb") as file:
            del self.tokenizer.text
            del self.tokenizer.scanner.text
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    def print_positional_index(self):
        for postings_list in self.dictionary.values():
            postings_list.print_postings()
