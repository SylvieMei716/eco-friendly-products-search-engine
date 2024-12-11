"""
This file is the pipeline to run the search.

Authors: Lindsey Dye, Sylvie Mei
"""
import os
import gzip
import jsonlines
import pickle
from collections import defaultdict
from tqdm import tqdm
import time
import requests
import random

from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType, BasicInvertedIndex
from ranker import BM25, Ranker, CrossEncoderScorer
from l2r import L2RRanker, L2RFeatureExtractor
from multimodal import MultimodalSearch
import csv

DATA_PATH = './data/'
CACHE_PATH = './__pycache__/'

STOPWORD_PATH = DATA_PATH + 'stopwords.txt'
DATASET_PATH = DATA_PATH + 'Beauty_and_Fashion.jsonl.gz'
DOCID_TO_IMAGE_PATH = CACHE_PATH + 'docid_to_image.pkl'
DOCID_TO_PRICE_PATH = CACHE_PATH + 'docid_to_price.pkl'
DOCID_TO_RATING_PATH = CACHE_PATH + 'docid_to_rating.pkl'
DOCID_TO_ECOLABEL_PATH = CACHE_PATH + 'docid_to_ecolabel.pkl'
DOCID_TO_LINK_PATH = CACHE_PATH + 'docid_to_link.pkl'
DOCID_TO_TITLE_PATH = CACHE_PATH + 'docid_to_title.pkl'
DOCID_TO_DESC_PATH = CACHE_PATH + 'docid_to_desc.pkl'
TITLE_INDEX = 'title_index'

class EcoSearchEngine:
    def __init__(self, max_docs: int = -1, use_l2r: bool = False, multimodal: bool = False):
        print("Initializing Eco-Friendly Search Engine...")
        
        # Load stopwords
        self.stopwords = set()
        with open(STOPWORD_PATH, 'r', encoding='utf-8') as f:
            self.stopwords = set(f.read().splitlines())

        # Tokenizer for preprocessing
        self.tokenizer = RegexTokenizer()

        # Indexing datasets
        self.index = BasicInvertedIndex()
        self.dataset = self.load_dataset(DATASET_PATH, max_docs)
        self.index_documents()
        
        self.title_index = BasicInvertedIndex()
        self.index_titles()

        # Initialize BM25 as the default ranker
        print("Initializing BM25 Ranker...")
        self.bm25_ranker = BM25(self.index)
        self.ranker = Ranker(self.index, self.tokenizer, self.stopwords, self.bm25_ranker)

        # Multimodal Search
        print("Initializing multimodal object...")
        self.multimodal = None
        if multimodal:
            self.multimodal = MultimodalSearch()
            
        print("Loading docid to image info...")
        self.docid_to_image = pickle.load(open(DOCID_TO_IMAGE_PATH, 'rb'))
        print("Loading docid to price info...")
        self.docid_to_price = pickle.load(open(DOCID_TO_PRICE_PATH, 'rb'))
        print("Loading docid to rating info...")
        self.docid_to_rating = pickle.load(open(DOCID_TO_RATING_PATH, 'rb'))
        print("loading docid to ecolabel info...")
        self.docid_to_ecolabel = pickle.load(open(DOCID_TO_ECOLABEL_PATH, 'rb'))
        print("Loading docid to link info")
        self.docid_to_link = pickle.load(open(DOCID_TO_LINK_PATH, 'rb'))
        print("Loading docid to title info...")
        self.docid_to_title = pickle.load(open(DOCID_TO_TITLE_PATH, 'rb'))
        print("Loading docid to description info...")
        self.docid_to_desc = pickle.load(open(DOCID_TO_DESC_PATH, 'rb'))
        
        with open("data/training_set.csv", 'r', encoding='cp850') as f:
            data = csv.reader(f)
            train_docs = set()
            for idx, row in tqdm(enumerate(data)):
                if idx == 0:
                    continue
                train_docs.add(row[2])
        
        self.raw_text_dict = defaultdict()
        file = gzip.open(DATASET_PATH, 'rt')
        with jsonlines.Reader(file) as reader:
            while True:
                try:
                    data = reader.read()
                    if str(data['docid']) in train_docs:
                        self.raw_text_dict[str(
                            data['docid'])] = data['text'][:500]
                except:
                    break
        pickle.dump(
            self.raw_text_dict,
            open(CACHE_PATH + 'raw_text_dict_train.pkl', 'wb')
        )
        self.cescorer = CrossEncoderScorer(self.raw_text_dict)

        # Learning-to-Rank
        self.l2r_ranker = None
        if use_l2r:
            self.l2r_feature_extractor = L2RFeatureExtractor(
                self.index, self.title_index, self.tokenizer, self.stopwords, self.cescorer, self.multimodal, doc_image_info=self.docid_to_image, 
                keywords='sustainable organic biodegradable recyclable compostable recycled non-toxic renewable plant-based vegan low-impact zero-waste green cruelty-free FSC-certified carbon-neutral Energy Star Fair Trade eco-conscious climate-positive upcycled responsibly sourced energy-efficient plastic-free pesticide-free natural ethical eco-label water-saving low-carbon toxin-free green-certified eco-safe'
            )
            self.l2r_ranker = L2RRanker(self.index, self.title_index, self.tokenizer, self.stopwords, self.ranker, self.l2r_feature_extractor)
            self.load_or_train_l2r()

        print("Eco-Friendly Search Engine Initialized!")

    def load_dataset(self, path, max_docs):
        """Load dataset from a JSONL.gz file."""
        dataset = []
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            reader = jsonlines.Reader(f)
            for idx, doc in enumerate(tqdm(reader, desc="Loading dataset")):
                if 0 < max_docs <= idx:
                    break
                dataset.append(doc)
        return dataset

    def index_documents(self):
        """Index documents using BasicInvertedIndex."""
        print("Indexing documents...")
        for doc in tqdm(self.dataset, desc="Indexing"):
            text = doc.get('description', '')
            tokens = self.tokenizer.tokenize(text)
            filtered_tokens = [t for t in tokens if t not in self.stopwords]
            self.index.add_doc(doc['docid'], filtered_tokens)
        print("Indexing complete.")
        
    def index_titles(self):
        """Index documents using BasicInvertedIndex."""
        print("Indexing titles...")
        for doc in tqdm(self.dataset, desc="Indexing"):
            text = doc.get('title', '')
            tokens = self.tokenizer.tokenize(text)
            filtered_tokens = [t for t in tokens if t not in self.stopwords]
            self.title_index.add_doc(doc['docid'], filtered_tokens)
        print("Indexing complete.")

    def check_amazon_item_exists(self, url):
        """Check if the item link is still valid."""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        time.sleep(random.uniform(2, 5))
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            if "currently unavailable" in response.text or "couldn't find that page" in response.text:
                return False
            else:
                return True
        elif response.status_code == 404:
            return False
        else:
            return False

    def search(self, query, sort_option="relevance"):
        """Perform a search query."""
        print(f"Searching for: {query}")
        tokens = self.tokenizer.tokenize(query)
        filtered_tokens = [t for t in tokens if t not in self.stopwords]
        results = self.ranker.query(" ".join(filtered_tokens))

        # Enrich results with metadata
        enriched_results = []
        for result in results:
            # doc = next((d for d in self.dataset if d['docid'] == result[0]), {})
            enriched_results.append({
                "docid": result[0],
                "title": self.docid_to_title.get(result[0], "Unknown"),
                "description": self.docid_to_desc.get(result[0], "No description available"),
                "score": result[1],
                "price": self.docid_to_price.get(result[0], "N/A"),
                "eco_friendly": str(self.docid_to_ecolabel.get(result[0], "Unknown")),
                "image": self.docid_to_image.get(result[0], ""),
                "avg_rating": self.docid_to_rating.get(result[0], "N/A"),
                "link": self.docid_to_link.get(result[0], "")
            })

        # Sort results by relevance and rating if requested
        if sort_option != "relevance":
            RATING_WEIGHT = 0.3
            enriched_results = sorted(enriched_results, key=lambda x: x["score"]*(1-RATING_WEIGHT) + x["avg_rating"]*0.1*RATING_WEIGHT, reverse=True)
        
        # Ensure the items in the first page have valid link
        print("Checking the link for the first page...")
        valid_url_cnt = 0
        for i in range(min(10, len(enriched_results))):
            result = enriched_results[i]
            if not self.check_amazon_item_exists(self.docid_to_link.get(result['docid'])):
                enriched_results.pop(i)
                valid_url_cnt += 1
                break
                
        
        return enriched_results

    def load_or_train_l2r(self):
        """Load or train the Learning-to-Rank model."""
        model_path = os.path.join(CACHE_PATH, 'l2r.pkl')
        # if os.path.exists(model_path):
        #     print("Loading pre-trained L2R model...")
        #     with open(model_path, 'rb') as f:
        #         self.l2r_ranker = pickle.load(f)
        # else:
        print("Training L2R model...")
        training_data = "data/training_set.csv"  # Replace with actual training file path
        self.l2r_ranker.train(training_data)
        with open(model_path, 'wb') as f:
            pickle.dump(self.l2r_ranker, f)

# Initialize function for app.py
def initialize():
    return EcoSearchEngine(max_docs=-1, use_l2r=True, multimodal=True)


def main():
    search_obj = EcoSearchEngine(max_docs=-1, use_l2r=True, multimodal=True)
    query = "women maxi dress"
    results = search_obj.search(query, sort_option="relevance")
    print(results[:5])


if __name__ == '__main__':
    main()