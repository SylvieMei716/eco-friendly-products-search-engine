'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
#from document_preprocessor import Tokenizer
from collections import Counter, defaultdict
import json
import os
import uuid
import glob
import jsonlines
import gzip
#from document_preprocessor import RegexTokenizer


class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter()  # token count
        self.vocabulary = set()  # the vocabulary of the collection
        # metadata like length, number of unique tokens of the documents
        self.document_metadata = {}

        self.index = defaultdict(list)  # the index

    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.index = defaultdict(list)
        self.document_metadata = {}
        self.statistics['unique_token_count'] = 0
        self.statistics['total_token_count'] = 0
        self.statistics['number_of_documents'] = 0
        self.statistics['mean_document_length'] = 0
        self.statistics['unique_tokens'] = set()
        self.term_metadata = {}
        self.termdoc_data = {}

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        if docid in self.document_metadata:
            uni_tokens, length, tokens = self.document_metadata[docid]
            u_tokens = set(tokens)
            del self.document_metadata[docid] 

            for token in u_tokens:
                    self.term_metadata[token]['term_count'] -= tokens.count(token)
                    self.term_metadata[token]['doc_frequency'] -= 1

            for token in u_tokens:
                l = self.get_postings(token)
                if len(l)==1 and docid in l[0]:
                    self.statistics['unique_token_count'] -= 1
                    self.statistics['unique_tokens'].remove(token)

            self.statistics['total_token_count'] -= length
            self.statistics['number_of_documents'] -= 1
            if self.statistics['number_of_documents'] != 0:
                self.statistics['mean_document_length'] = self.statistics['total_token_count']/self.statistics['number_of_documents']
            else:
                self.statistics['mean_document_length'] = self.statistics['mean_document_length']

        else:
            print('The document does not exist')
        return None
        
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        if docid not in self.document_metadata:
            token_count = len(tokens)
            self.document_metadata[docid] = (0, 0, tokens)
            self.statistics['total_token_count'] += token_count
            self.statistics['number_of_documents'] += 1

            token_counts = Counter(tokens)
            self.termdoc_data[docid] = token_counts

            for token in set(tokens):
                if token not in self.term_metadata:
                    self.term_metadata[token] = {'term_count':token_counts[token],'doc_frequency':1}
                else:
                    self.term_metadata[token]['term_count'] += token_counts[token]
                    self.term_metadata[token]['doc_frequency'] += 1

            new_unique_tokens = set(tokens) - self.statistics['unique_tokens']
            self.statistics['unique_tokens'].update(new_unique_tokens)
            self.statistics['unique_token_count'] += len(new_unique_tokens)
            #print(len(new_unique_tokens))

            if self.statistics['number_of_documents'] > 0:
                self.statistics['mean_document_length'] = (
                    self.statistics['total_token_count'] / self.statistics['number_of_documents']
                )

        return None
    

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        postings = []

        # Iterate over document metadata
        for doc_id, metadata in self.document_metadata.items():
            _, _, tokens = metadata
            token_counts = Counter(tokens)
            if term in token_counts:
                postings.append((doc_id, token_counts[term]))

        return postings

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        dic = {}
        if doc_id in self.document_metadata:
            u,l, tokens = self.document_metadata[doc_id]
            u_tokens = set(tokens)
            u += len(u_tokens)
            l += len(tokens)
            dic = {'unique_tokens':u, 'length':l}
        return dic


    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        """dic = {}
        info = self.get_postings(term)
        if len(info)>0:
            count = 0
            freq = len(info)
            for i in range(len(info)):
                doc, t = info[i]
                count += t
            dic = {'term_count':count,'doc_frequency':freq}"""
        dic = self.term_metadata[term]
        return dic

    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        dic = {}
        self.statistics['unique_token_count'] = len(self.statistics['unique_tokens'])
        dic['unique_token_count'] = self.statistics['unique_token_count']
        dic['total_token_count'] = self.statistics['total_token_count']
        dic['number_of_documents'] = self.statistics['number_of_documents']
        dic['mean_document_length'] = self.statistics['mean_document_length']
        return dic
    
    def get_u_tokens(self):
        return self.statistics['unique_tokens']

    def save(self,index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        lis = []
        dirname = './' + index_directory_name
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        t = uuid.uuid4()
        filename = dirname + '/' + str(t) + '.txt'
        self.statistics['unique_tokens'] = list(self.statistics['unique_tokens'])
        lis.append(self.document_metadata) 
        lis.append(self.statistics)
        lis.append(self.term_metadata)
        with open(filename, 'w') as fd:
            fd.write(json.dumps(lis))
        return None

    def load(self,index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        dirname = './' + index_directory_name + '/*'
        file_list = glob.glob(dirname) 
        latest_file = max(file_list, key=os.path.getctime)
        with open(latest_file, 'r') as fd:
            lis = json.load(fd)
        self.document_metadata = lis[0]
        self.statistics = lis[1]
        self.term_metadata = lis[2]
        self.statistics['unique_tokens'] = set(self.statistics['unique_tokens'])
        return None


class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        self.statistics['index_type'] = 'PositionalInvertedIndex'
        self.index = defaultdict(list)
        self.document_metadata = {}
        self.statistics['unique_token_count'] = 0
        self.statistics['total_token_count'] = 0
        self.statistics['number_of_documents'] = 0
        self.statistics['mean_document_length'] = 0
        self.statistics['unique_tokens'] = set()

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        lis = []
        for doc in self.document_metadata:
            u,l,tokens = self.document_metadata[doc]
            c = 0
            p = [] 
            for i,token in enumerate(tokens):
                if token == term:
                    c += 1
                    p.append(i)
            if c!=0: lis.append((doc,c,p))

        return lis
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        #dic = {}
        """info = self.get_postings(term)
        if len(info)>0:
            count = 0
            freq = len(info)
            for i in range(len(info)):
                doc, t, p = info[i]
                count += t
            dic = {'term_count':count,'doc_frequency':freq}"""
        dic = self.term_metadata[term]
        return dic


class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = None, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the entire corpus at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index

        '''
         # TODO (HW3): This function now has an optional argument doc_augment_dict; check README.md
       
        # HINT: Think of what to do when doc_augment_dict exists, how can you deal with the extra information?
        #       How can you use that information with the tokens?
        #       If doc_augment_dict doesn't exist, it's the same as before, tokenizing just the document text
          
        # TODO: Implement this class properly. This is responsible for going through the documents
        #       one by one and inserting them into the index after tokenizing the document

        # TODO: Figure out what type of InvertedIndex to create.
        #       For HW3, only the BasicInvertedIndex is required to be supported

        # TODO: If minimum word frequencies are specified, process the collection to get the
        #       word frequencies

        # NOTE: Make sure to support both .jsonl.gz and .jsonl as input
                      
        # TODO: Figure out which set of words to not index because they are stopwords or
        #       have too low of a frequency

        # TODO: Read the collection and process/index each document.
        #       Only index the terms that are not stopwords and have high-enough frequency

        if not text_key:
            text_key = 'text'
        def filter_stopwords(tokens, stopwords):
            return [token for token in tokens if token not in stopwords]

        def filter_by_frequency(tokens, min_freq, u_tokens):
            token_counts = Counter(tokens)
            return [token for token in u_tokens if token_counts[token] >= min_freq]
        
        if index_type.name == 'BasicInvertedIndex':
            index = BasicInvertedIndex()
        elif index_type.name == 'PositionalIndex':
            index = PositionalInvertedIndex()

        if '.gz' in dataset_path:
            with gzip.open(dataset_path,'rt') as file:
                for line in file:
                    data = json.loads(line)
        else:
            with jsonlines.open(dataset_path) as file:
                data = list(file.iter())

        tokenizer = document_preprocessor
        all_tokens = []

        if not max_docs:
            max_docs = len(data)

        if doc_augment_dict:
            for i in range(max_docs):
                tokens = tokenizer.tokenize(data[i][text_key])

                if data[i]["docid"] in doc_augment_dict:
                    for query in doc_augment_dict[data[i]["docid"]]:
                        queries = tokenizer.tokenize(query)
                        tokens.extend(queries)

                index.add_doc(data[i]["docid"], tokens)
                #print('added docs')
                all_tokens.extend(tokens)
        else:
            for i in range(max_docs):
                tokens = tokenizer.tokenize(data[i][text_key])
                index.add_doc(data[i]["docid"], tokens)
                #print('added docs')
                all_tokens.extend(tokens) 

        unique_tokens = index.get_u_tokens()

        if stopwords:
            unique_tokens = filter_stopwords(unique_tokens, stopwords)

        if minimum_word_frequency > 0:
            unique_tokens = filter_by_frequency(all_tokens, minimum_word_frequency, unique_tokens)

        index.statistics['unique_tokens'] = list(unique_tokens)

        return index


'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1

    def save(self):
        print('Index saved!')

if __name__=="__main__":
    """index = Indexer.create_index(IndexType.BasicInvertedIndex, './tests/dataset_2.jsonl', RegexTokenizer('\w+'), set(), 0)
    stats = index.get_statistics()
    print(stats)"""
    pass
