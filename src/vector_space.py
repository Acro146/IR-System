import pickle
from io import StringIO
from math import log10, log, sqrt
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from texttable import Texttable

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
    cosin_similarity: float
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
        # document_length initialization with 0 for each document
        self.document_length = OrderedDict.fromkeys(self.document_collection, 0)
        self.calculate_terms_schemes(positional_index)
        self.terms_in_documents = OrderedDict(sorted(self.terms_in_documents.items(), key=lambda x: x[0]))

    def calculate_terms_schemes(self, positional_index):
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

    def get_query_results(self, phrase, document):
        phrase = self.tokenizer(StringIO(phrase))
        query_result = QueryResult(terms=OrderedDict(), length=0, cosin_similarity=0,
                                   query_document_product=OrderedDict())
        query_result = self.__get_terms_query_data(query_result, phrase)
        query_result = self.__calculate_terms_query_tf_idf(query_result)
        query_result = self.__calculate_query_length(query_result)
        query_result = self.__calculate_cosin_similarity(query_result, document)
        return query_result

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

    def __calculate_terms_query_tf_idf(self, query_result):
        for term, term_data in query_result.terms.items():
            term_data.tf_idf = term_data.w_tf * term_data.idf
        return query_result

    def __calculate_query_length(self, query_result):
        for term_data in query_result.terms.values():
            query_result.length += term_data.tf_idf ** 2
        query_result.length = sqrt(query_result.length)
        return query_result

    def __calculate_cosin_similarity(self, query_result, document):
        for term, term_data in query_result.terms.items():
            if term in self.terms_in_documents.keys():
                term_data.norm_tf_idf = term_data.tf_idf / query_result.length if query_result.length else 0
                query_result.query_document_product.update(
                    {term: {document.document_path.name: term_data.norm_tf_idf * self.terms_in_documents[term].norm_tf_idf[document]}}
                )
                query_result.cosin_similarity += query_result.query_document_product[term][document.document_path.name]
            else:
                query_result.query_document_product.update(
                    {term: {document.document_path.name: 0}}
                )
        return query_result

    def print_vector_space(self):
        mx_width = 200
        precision = 9

        print('-' * 50 + "Vector Space" + '-' * 50)

        self.__print_table_td("Term Frequency", "tf")
        # -----------------------------
        self.__print_table_td("w tf(1+ log tf)", "w_tf")
        # -----------------------------
        print('\n\n')
        df_idf_table = Texttable(max_width=mx_width).set_precision(precision)
        print('-' * 50 + "Document Frequency" + '-' * 50)
        df_idf_table.add_row(["Term", "DF", "IDF"])

        for term, term_data in self.terms_in_documents.items():
            df_idf_table.add_row([term, term_data.df, term_data.idf])

        print(df_idf_table.draw())
        # -----------------------------

        self.__print_table_td("TF-IDF", "tf_idf")
        # -----------------------------
        print('\n\n')
        print('-' * 50 + "Document length" + '-' * 50)
        doc_len_table = Texttable(max_width=mx_width).set_precision(precision)
        doc_len_table.add_row(["Document", "Length"])
        for doc in self.document_collection:
            doc_len_table.add_row([doc.document_path.name, self.document_length[doc]])
        print(doc_len_table.draw())

        # -----------------------------

        self.__print_table_td("Normalized tf*idf", "norm_tf_idf")

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

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)

    def save(self, file_name):
        with open(Path(self.document_collection.directory) / file_name, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
