# class Ranker:
#     """
#     The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
#     using a particular relevance function (e.g., BM25).
#     A Ranker can be configured with any RelevanceScorer.
#     """

#     def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
#                  scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
#         """
#         Initializes the state of the Ranker object.

#         Args:
#             index: An inverted index
#             document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
#             stopwords: The set of stopwords to use or None if no stopword filtering is to be done
#             scorer: The RelevanceScorer object
#             raw_text_dict: A dictionary mapping document ids to their raw text. Optional.
#         """
#         self.index = index
#         self.tokenize = document_preprocessor.tokenize
#         self.scorer = scorer
#         self.stopwords = stopwords
#         self.raw_text_dict = raw_text_dict

#     def query(self, query: str) -> list[tuple[int, float]]:
#         """
#         Searches the collection for relevant documents to the query and
#         returns a list of documents ordered by their relevance (most relevant first).

#         Args:
#             query: The query to search for

#         Returns:
#             A sorted list containing tuples of the document id and its relevance score
#         """
#         # Tokenize and count the frequency of each token in the query
#         query_tokens = self.tokenize(query)
#         query_word_counts = Counter(query_tokens)

#         # Remove stopwords if specified
#         if self.stopwords:
#             query_tokens = [token for token in query_tokens if token not in self.stopwords]
#             query_word_counts = Counter(query_tokens)

#         # Convert Counter to dict for further processing
#         query_word_counts = dict(query_word_counts)

#         # Initialize dictionaries to store document scores and word counts
#         doc_scores = {}
#         doc_word_counts = {}

#         # Iterate over each token in the query
#         for token in query_word_counts:
#             # Retrieve postings for each token from the index
#             postings = self.index.get_postings(token)
#             if postings:
#                 for docid, freq in postings:
#                     # Initialize document score and word count if not already present
#                     if docid not in doc_scores:
#                         doc_scores[docid] = 0
#                     if docid not in doc_word_counts:
#                         doc_word_counts[docid] = Counter()
                    
#                     # Update word count for the document and add to the document score
#                     doc_word_counts[docid][token] += freq
#                     doc_scores[docid] += self.scorer.score(docid, doc_word_counts[docid], query_word_counts)

#         # If no scores were computed (i.e., no query term matches any document), fallback to naive ranking
#         if not doc_scores:
#             doc_scores = self.naive_rank(query_tokens)

#         # Sort documents by their score (highest to lowest)
#         ranked_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

#         # Return the sorted list as a list of tuples (document id, relevance score)
#         return [(docid, score) for docid, score in ranked_docs]

#     def naive_rank(self, query_tokens: List[str]) -> dict[int, float]:
#         """
#         Naively ranks documents based on term frequency for the given query tokens.
        
#         Args:
#             query_tokens: The tokens in the query

#         Returns:
#             A dictionary of document ids and their scores based on term frequency
#         """
#         doc_scores = {}
        
#         for token in query_tokens:
#             postings = self.index.get_postings(token)
#             if postings:
#                 for docid, freq in postings:
#                     if docid not in doc_scores:
#                         doc_scores[docid] = 0
#                     doc_scores[docid] += freq
        
#         return doc_scores

"""
This file can be ran with different queries after implementing the previous code to the Ranker class in ranker.py.

"""

from indexing import InvertedIndex
import os
import pandas as pd
from collections import Counter
import gzip
import json
import jsonlines
from document_preprocessor import RegexTokenizer
from ranker import Ranker, BM25
from indexing import BasicInvertedIndex, IndexType
from relevance import run_relevance_tests

# Path to the combined CSV with all queries and relevance judgments
filename = './all_queries.csv'

# Create eco-friendly index
class eco_friendly_indexer:

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor, stopwords: set[str],
                     minimum_word_frequency: int, keywords: set[str],
                     max_docs: int = None) -> InvertedIndex:

        def filter_stopwords(tokens, stopwords):
            return [token for token in tokens if token not in stopwords]

        def filter_by_frequency(tokens, min_freq, u_tokens):
            token_counts = Counter(tokens)
            return [token for token in u_tokens if token_counts[token] >= min_freq]
        
        index = BasicInvertedIndex()

        if '.gz' in dataset_path:
            with gzip.open(dataset_path, 'rt') as file:
                data = [json.loads(line) for line in file]
        else:
            with jsonlines.open(dataset_path) as file:
                data = list(file.iter())

        tokenizer = document_preprocessor
        all_tokens = []

        if not max_docs:
            max_docs = len(data)

        for i in range(max_docs):
            tokens = tokenizer.tokenize(data[i]['description'])
            if any(keyword in tokens for keyword in keywords):
                index.add_doc(data[i]["docid"], tokens)
                all_tokens.extend(tokens)

        unique_tokens = index.get_tokens()

        if stopwords:
            unique_tokens = filter_stopwords(unique_tokens, stopwords)

        if minimum_word_frequency > 0:
            unique_tokens = filter_by_frequency(all_tokens, minimum_word_frequency, unique_tokens)

        index.statistics['unique_tokens'] = list(unique_tokens)

        return index

# Configuration
keywords_string = "sustainable, organic, biodegradable, recyclable, compostable, recycled, toxic, renewable, vegan, cruelty, FSC-certified, carbon, Fair Trade, climate, upcycled, responsibly sourced, pesticide, ethical, toxin, eco"
tokenizer = RegexTokenizer('\w+')
keywords = tokenizer.tokenize(keywords_string)

path = './data/Beauty_and_Fashion.jsonl.gz'  # Path to the dataset
stopwords_file = './data/stopwords.txt'
stopwords = set()
with open(stopwords_file, 'r') as f:
    for line in f:
        stopwords.add(line.strip())

eco_index = eco_friendly_indexer.create_index(IndexType.BasicInvertedIndex, path, RegexTokenizer('\w+'), stopwords, 3, set(keywords), max_docs=493293)

# Initialize the relevance scorer (BM25)
eco_bm25_scorer = BM25(eco_index)

# Initialize the ranker
eco_ranker = Ranker(eco_index, tokenizer, stopwords, eco_bm25_scorer)

# Define your queries
beauty_queries = ["sustainable makeup cerulean blue liquid eyeliner"
]

fashion_queries = ["handmade recycled purse"
]

# Evaluate Ranker function
def evaluate_ranker(ranker, queries):
    for query in queries:
        print(f'Running query: {query}')
        # Get BM25 scores with fallback to naive ranking
        bm25_doc_scores = ranker.query(query)
        # Run naive ranking independently
        query_tokens = ranker.tokenize(query)
        naive_doc_scores = ranker.naive_rank(query_tokens)
        # Format BM25 and naive scores for comparison
        print(f'Top documents and scores (BM25 with Naive Fallback):')
        # Iterate through BM25 results and print the naive fallback score if available
        for doc_id, bm25_score in bm25_doc_scores[:10]:
            naive_score = naive_doc_scores.get(doc_id, 0)  # Fallback to 0 if doc_id not in naive scores
            print(f'DocID: {doc_id}, BM25 Score: {bm25_score:.4f}, Naive Score: {naive_score}')
        print('---')

# Run evaluations
print("Evaluating BM25 Ranker with Naive Ranking Fallback:")
evaluate_ranker(eco_ranker, beauty_queries)
evaluate_ranker(eco_ranker, fashion_queries)

# Evaluate ranker with relevance tests
d = run_relevance_tests(filename, eco_ranker)
print(d)

