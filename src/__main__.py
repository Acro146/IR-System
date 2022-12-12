import os
import sys
from io import StringIO
from pathlib import Path

from src.preprocess import Normalizer, TokensScanner, Tokenizer, StopWords
from src.document_collection import DocumentCollection
from src.positional_index import PositionalIndex
from src.vector_space import VectorSpace


def index(collection):
    scanner = TokensScanner()
    stop_list = StopWords({'in', 'to', 'where'})
    normalizer = Normalizer()
    tokenizer = Tokenizer(stop_list, normalizer, scanner)
    document_collection = DocumentCollection(collection)

    positional_index = PositionalIndex(tokenizer, document_collection)
    vector_space = VectorSpace(tokenizer, positional_index)
    positional_index.save("index")
    vector_space.save("vector_space")


def print_index(collection):
    path = Path(collection) / "index"
    is_valid_file(path)
    positional_index = PositionalIndex.load(path)
    positional_index.print_positional_index()


def print_vector_space(collection):
    path = Path(collection) / "vector_space"
    is_valid_file(path)
    vector_space = VectorSpace.load(path)
    vector_space.print_vector_space()


def query(phrase, collection):
    positional_index_path = Path(collection) / "index"
    vector_space_path = Path(collection) / "vector_space"
    if not is_valid_file(positional_index_path) or not is_valid_file(vector_space_path):
        print("The 'index' or 'vector_space' are not found, please build the index first.")
        return
    positional_index = PositionalIndex.load(positional_index_path)
    vector_space = VectorSpace.load(vector_space_path)
    matched_documents = positional_index.phrase_query(StringIO(phrase))
    vector_space.query(StringIO(phrase), matched_documents)


def is_valid_file(path):
    return os.path.exists(path)


def main():
    print("Please enter number of the command you want to execute:")
    select = input("[1] Build Positional Index\n"
                   "[2] Print Positional Index\n"
                   "[3] Print Vector Space\n"
                   "[4] Phrase Query\n"
                   "[5] Exit: ")

    if select == "1":
        collection = input("Enter collection path: ")
        if not is_valid_file(collection):
            print("The collection not found!")
            return
        index(collection)

    elif select == "2":
        collection = input("Enter collection path: ")
        if not is_valid_file(collection):
            print("The collection not found!")
            return
        if not is_valid_file(Path(collection) / "vector_space"):
            print("The index not found, please build the index first.")
            return
        print_index(collection)

    elif select == "3":
        collection = input("Enter collection path: ")
        if not is_valid_file(collection):
            print("The collection not found!")
            return
        if not is_valid_file(Path(collection) / "index") or not is_valid_file(Path(collection) / "vector_space"):
            print("The 'index' or 'vector_space' are not found, please build the index first.")
            return
        print_vector_space(collection)

    elif select == "4":
        collection = input("Enter collection path: ")
        if not is_valid_file(collection):
            print("The collection not found!")
            return
        if not is_valid_file(Path(collection) / "index") or not is_valid_file(Path(collection) / "vector_space"):
            print("The 'index' or 'vector_space' are not found, please build the index first.")
            return
        phrase = input("Enter phrase: ")
        query(phrase, collection)

    else:
        sys.exit()


if __name__ == '__main__':
    main()
