import os
import sys
from collections import OrderedDict
from io import StringIO
from pathlib import Path
from texttable import Texttable

from src.preprocess import Normalizer
from src.preprocess import TokensScanner
from src.preprocess import StopWords
from src.preprocess import Tokenizer
from src.document_collection import DocumentCollection
from src.positional_index import PositionalIndex, Candidate
from src.vector_space import VectorSpace

Collection_Path = "../Document Collection"


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
    is_valid_file(positional_index_path)
    is_valid_file(vector_space_path)
    positional_index = PositionalIndex.load(positional_index_path)
    vector_space = VectorSpace.load(vector_space_path)

    matched_documents = positional_index.phrase_query(StringIO(phrase))

    ranked_result: OrderedDict[Candidate, float]
    ranked_result = OrderedDict()

    query_result: list[vector_space.QueryResult()]
    query_result = []
    for candidate in matched_documents:
        result = vector_space.get_query_results(phrase, candidate.document)
        if not len(query_result):
            query_result.append(result)
        ranked_result.update({candidate: result.cosin_similarity})

    print_query_result(query_result)
    print_ranked_result(ranked_result)


def print_query_result(query_result):
    print('*' * 50 + "\tQuery Result\t" + '*' * 50)
    query_result_table = Texttable()
    query_result_table.add_row(["term", "tf", "df", "idf", "w_tf", "tf-idf", "norm_tf-idf"])
    for result in query_result:
        for term, term_data in result.terms.items():
            query_result_table.add_row(
                [term, term_data.tf, term_data.df, term_data.idf, term_data.w_tf, term_data.tf_idf,
                 term_data.norm_tf_idf])
    print(query_result_table.draw())


def print_ranked_result(ranked_result):
    ranked_table = Texttable()
    print('*' * 50 + "\tRanked Result\t" + '*' * 50)
    ranked_result = OrderedDict(sorted(ranked_result.items(), key=lambda item: item[1], reverse=True))
    ranked_table = Texttable(max_width=200).set_precision(9)
    ranked_table.add_rows([["Document", "Position", "cosin_similarity"]])
    for doc, cosin_similarity in ranked_result.items():
        ranked_table.add_row([doc.document.document_path.name, doc.position, cosin_similarity])
    print(ranked_table.draw())


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
        query(input("Enter the phrase: "), collection)

    else:
        sys.exit()


if __name__ == '__main__':
    main()
