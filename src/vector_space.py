import pickle
from math import log10, log, sqrt
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List
from texttable import Texttable

from .positional_index import Candidate
from .document_collection import Document, DocumentCollection
from .preprocess import Tokenizer


@dataclass
class TermDocumentsData:
    tf: OrderedDict[Document, int]
    df: int
    idf: float
    tf_idf: OrderedDict[Document, float]
    w_tf: OrderedDict[Document, float]
    norm_tf_idf: OrderedDict[Document, float]


@dataclass
class TermData:
    tf: int
    df: int
    idf: float
    w_tf: float
    tf_idf: float
    norm_tf_idf: float


@dataclass
class QueryResult:
    terms: OrderedDict[str, TermData]
    length: float
    cosine_similarity: float
    query_document_product: OrderedDict[str, OrderedDict[str, float]]


@dataclass
class VectorSpace:
    document_collection: DocumentCollection
    tokenizer: Tokenizer
    terms_in_documents: OrderedDict[str, TermDocumentsData]
    document_length: OrderedDict[Document, float]

    def __init__(self, tokenizer, positional_index):
        self.document_collection = positional_index.document_collection
        self.tokenizer = tokenizer
        self.terms_in_documents = OrderedDict()
        self.document_length = OrderedDict.fromkeys(self.document_collection, 0)
        self.__calculate_terms_schemes(positional_index)

    def __calculate_terms_schemes(self, positional_index):
        number_of_documents = positional_index.number_of_documents

        for term, postings_list in positional_index.dictionary.items():
            term_data = TermDocumentsData(tf=OrderedDict(), df=len(postings_list.postings),
                                          idf=log10(number_of_documents / len(postings_list.postings)),
                                          tf_idf=OrderedDict(),
                                          w_tf=OrderedDict(),
                                          norm_tf_idf=OrderedDict())

            for posting in postings_list.postings:
                term_data.tf[posting.document] = len(posting.positions)
                term_data.tf_idf[posting.document] = term_data.tf[posting.document] * term_data.idf
                term_data.w_tf[posting.document] = 1.0 + log(term_data.tf[posting.document])
                self.document_length[posting.document] = self.document_length[posting.document] + term_data.tf_idf[
                    posting.document] ** 2

            for document in self.document_collection:
                if document not in term_data.tf:
                    term_data.tf.update({document: 0})
                    term_data.tf_idf.update({document: 0})
                    term_data.w_tf.update({document: 0})
                    term_data.norm_tf_idf.update({document: 0})

            term_data.tf = OrderedDict(sorted(term_data.tf.items(), key=lambda x: x[0].document_id))
            self.terms_in_documents.update({term: term_data})

        for document in self.document_collection:
            self.document_length[document] = sqrt(self.document_length[document])

        for term, term_data in self.terms_in_documents.items():
            for document in self.document_collection:
                term_data.norm_tf_idf[document] = term_data.tf_idf[document] / self.document_length[document] if \
                    self.document_length[document] else 0
            term_data.norm_tf_idf = OrderedDict(sorted(term_data.norm_tf_idf.items(), key=lambda x: x[0].document_id))

    def print_vector_space(self):
        mx_width = 200
        precision = 9
        print('-' * 50 + "Vector Space" + '-' * 50)

        self.__print_table_td("Term Frequency", "tf")

        self.__print_table_td("w_tf(1+ log tf)", "w_tf")

        print('\n\n')
        df_idf_table = Texttable(max_width=mx_width).set_precision(precision)
        print('-' * 50 + "Document Frequency" + '-' * 50)
        df_idf_table.add_row(["Term", "DF", "IDF"])
        for term, term_data in self.terms_in_documents.items():
            df_idf_table.add_row([term, term_data.df, term_data.idf])
        print(df_idf_table.draw())

        self.__print_table_td("TF-IDF (w_tf*idf)", "tf_idf")

        print('\n\n')
        print('-' * 50 + "Document length (sqrt(sum(tf-idf**2)))" + '-' * 50)
        doc_len_table = Texttable(max_width=mx_width).set_precision(precision)
        doc_len_table.add_row(["Document", "Length"])
        for doc in self.document_collection:
            doc_len_table.add_row([doc.document_path.name, self.document_length[doc]])
        print(doc_len_table.draw())

        self.__print_table_td("Normalized tf-idf (tf-idf/document_length)", "norm_tf_idf")

    def __print_table_td(self, table_name, data):
        print('\n\n')
        print('-' * 50 + table_name + '-' * 50)
        table = Texttable(max_width=200).set_precision(9)
        table_raw = ["Term"]
        table_raw.extend(
            [str(f"{doc.document_path.name}\n id:{doc.document_id}") for doc in self.document_collection])
        table.add_row(table_raw)
        for term, term_data in self.terms_in_documents.items():
            table_raw = [term]
            table_raw.extend([self.__switch(data, term_data)[doc] for doc in self.document_collection])
            table.add_row(table_raw)
        print(table.draw())

    @classmethod
    def __switch(cls, data, term_data):
        switcher = {
            "tf": term_data.tf,
            "w_tf": term_data.w_tf,
            "df": term_data.df,
            "idf": term_data.idf,
            "tf_idf": term_data.tf_idf,
            "norm_tf_idf": term_data.norm_tf_idf,
        }
        return switcher.get(data, "Invalid data")

    def query(self, phrase, candidates):
        query_phrase = list(self.tokenizer(phrase))
        query_results: List[(str, QueryResult)] = []
        query_result: QueryResult

        for candidate in candidates:
            query_result = QueryResult(terms=OrderedDict(), length=0, cosine_similarity=0,
                                       query_document_product=OrderedDict())
            query_result = self.__get_terms_query_data(query_result, query_phrase)
            query_result = self.__calculate_terms_query_tf_idf(query_result)

            query_result = self.__calculate_query_length(query_result)
            query_result = self.__calculate_cosine_similarity(query_result, candidate.document)
            query_results.append((candidate, query_result))

        if query_results:
            self.__print_query_terms_data(query_results[0][1])
            self.__print_query_document_product(query_results, query_phrase)
            self.__print_query_results(query_results)

    def __get_terms_query_data(self, query_result, phrase):
        for term in phrase:
            if term in query_result.terms:
                query_result.terms[term].tf += 1
                query_result.terms[term].w_tf = 1.0 + log(query_result.terms[term].tf)
                continue
            if term in self.terms_in_documents:
                query_result.terms.update(
                    {term: TermData(tf=1,
                                    df=self.terms_in_documents[term].df,
                                    idf=self.terms_in_documents[term].idf,
                                    w_tf=1,
                                    tf_idf=0,
                                    norm_tf_idf=0)})
            else:
                query_result.terms.update(
                    {term: TermData(tf=1,
                                    df=0,
                                    idf=0,
                                    w_tf=1,
                                    tf_idf=0,
                                    norm_tf_idf=0)})

        return query_result

    @classmethod
    def __calculate_terms_query_tf_idf(cls, query_result):
        for term_data in query_result.terms.values():
            term_data.tf_idf = term_data.w_tf * term_data.idf
        return query_result

    @classmethod
    def __calculate_query_length(cls, query_result):
        for term_data in query_result.terms.values():
            query_result.length += term_data.tf_idf ** 2
        query_result.length = sqrt(query_result.length)
        return query_result

    def __calculate_cosine_similarity(self, query_result, document):
        for term, term_data in query_result.terms.items():
            if term in self.terms_in_documents.keys():
                term_data.norm_tf_idf = term_data.tf_idf / query_result.length if query_result.length else 0
                query_result.query_document_product.update(
                    {term: {
                        document.document_path.name: term_data.norm_tf_idf * self.terms_in_documents[term].norm_tf_idf[
                            document]}}
                )
                query_result.cosine_similarity += query_result.query_document_product[term][document.document_path.name]
            else:
                query_result.query_document_product.update(
                    {term: {document.document_path.name: 0}}
                )
        return query_result

    def __print_query_results(self, query_results):
        ranked_result: OrderedDict[Candidate, float]
        ranked_result = OrderedDict()
        for candidate, result in query_results:
            ranked_result.update({candidate: result.cosine_similarity})
        self.__print_ranked_result(ranked_result)

    @classmethod
    def __print_query_terms_data(cls, query_result):
        print('*' * 50 + "\tQuery Result\t" + '*' * 50)
        query_result_table = Texttable(max_width=200).set_precision(9)
        query_result_table.add_row(["term", "tf", "df", "idf", "w_tf", "tf-idf", "norm_tf-idf"])

        for term, term_data in query_result.terms.items():
            query_result_table.add_row(
                [term, term_data.tf, term_data.df, term_data.idf, term_data.w_tf, term_data.tf_idf,
                 term_data.norm_tf_idf])
        print(query_result_table.draw())

    @classmethod
    def __print_ranked_result(cls, ranked_result):
        print('*' * 50 + "\tRanked Result\t" + '*' * 50)
        ranked_result = OrderedDict(sorted(ranked_result.items(), key=lambda item: item[1], reverse=True))
        ranked_table = Texttable(max_width=200).set_precision(9)
        ranked_table.add_rows([["Document", "Position", "cosine_similarity"]])
        for doc, cosine_similarity in ranked_result.items():
            ranked_table.add_row([doc.document.document_path.name, doc.position, cosine_similarity])
        print(ranked_table.draw())

    def __print_query_document_product(self, query_results, phrase):
        terms_documents_product: OrderedDict[str, OrderedDict[str, float]]
        terms_documents_product = OrderedDict()
        for term in phrase:
            terms_documents_product.update({term: OrderedDict()})
            for query_result in query_results:
                terms_documents_product[term].update(query_result[1].query_document_product[term])

        print('-' * 50 + "Query Document Product" + '-' * 50)
        table = Texttable(max_width=200).set_precision(9)
        table_raw = ["Term"]
        table_raw.extend([str(f"{doc.document_path.name}\n id:{doc.document_id}") for doc in self.document_collection])
        table.add_row(table_raw)

        for term, term_data in terms_documents_product.items():
            table_raw = [term]
            for doc in self.document_collection:
                if doc.document_path.name in term_data.keys():
                    table_raw.append(term_data[doc.document_path.name])
                else:
                    table_raw.append(0)
            table.add_row(table_raw)
        print(table.draw())


    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    def save(self, file_name):
        with open(Path(self.document_collection.directory) / file_name, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
