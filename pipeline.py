import os
import gzip
import jsonlines
import pickle
from collections import defaultdict
from tqdm import tqdm

from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType, BasicInvertedIndex
from ranker import BM25, Ranker
from l2r import L2RRanker, L2RFeatureExtractor
from multimodal import MultimodalSearch

DATA_PATH = './data/'
CACHE_PATH = './__cache__/'

STOPWORD_PATH = DATA_PATH + 'stopwords.txt'
DATASET_PATH = DATA_PATH + 'meta_All_Beauty.jsonl.gz'

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

        # Initialize BM25 as the default ranker
        self.bm25_ranker = BM25(self.index)
        self.ranker = Ranker(self.index, self.tokenizer, self.stopwords, self.bm25_ranker)

        # Multimodal Search
        self.multimodal = None
        if multimodal:
            self.multimodal = MultimodalSearch()

        # Learning-to-Rank
        self.l2r_ranker = None
        if use_l2r:
            self.l2r_feature_extractor = L2RFeatureExtractor(
                self.index, None, None, self.tokenizer, self.stopwords, {}, {}
            )
            self.l2r_ranker = L2RRanker(self.index, None, self.tokenizer, self.stopwords, self.ranker, self.l2r_feature_extractor)
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
            text = doc.get('title', '') + ' ' + doc.get('description', '')
            tokens = self.tokenizer.tokenize(text)
            filtered_tokens = [t for t in tokens if t not in self.stopwords]
            self.index.add_document(doc['asin'], filtered_tokens)
        print("Indexing complete.")

    def search(self, query, sort_by_price=False):
        """Perform a search query."""
        print(f"Searching for: {query}")
        tokens = self.tokenizer.tokenize(query)
        filtered_tokens = [t for t in tokens if t not in self.stopwords]
        results = self.ranker.rank(filtered_tokens)

        # Enrich results with metadata
        enriched_results = []
        for result in results:
            doc = next((d for d in self.dataset if d['asin'] == result[0]), {})
            enriched_results.append({
                "id": result[0],
                "title": doc.get("title", "Unknown"),
                "description": doc.get("description", "No description available"),
                "score": result[1],
                "price": doc.get("price", "N/A"),
                "eco_friendly": self.tag_eco_friendly(doc),
                "image": doc.get("image", ""),
            })

        # Sort results by price if requested
        if sort_by_price:
            enriched_results = sorted(enriched_results, key=lambda x: x.get("price", float('inf')))
        
        return enriched_results

    def tag_eco_friendly(self, doc):
        """Tag products as eco-friendly or not based on keywords."""
        eco_keywords = ['sustainable', 'organic', 'biodegradable', 'recyclable', 'compostable']
        non_eco_keywords = ['non-recyclable', 'disposable', 'single-use']
        title = doc.get("title", "").lower()
        description = doc.get("description", "").lower()
        
        for keyword in eco_keywords:
            if keyword in title or keyword in description:
                return "Eco-Friendly"
        for keyword in non_eco_keywords:
            if keyword in title or keyword in description:
                return "Not Eco-Friendly"
        return "Uncertain"

    def load_or_train_l2r(self):
        """Load or train the Learning-to-Rank model."""
        model_path = os.path.join(CACHE_PATH, 'l2r.pkl')
        if os.path.exists(model_path):
            print("Loading pre-trained L2R model...")
            with open(model_path, 'rb') as f:
                self.l2r_ranker = pickle.load(f)
        else:
            print("Training L2R model...")
            training_data = "data/training_data.csv"  # Replace with actual training file path
            self.l2r_ranker.train(training_data)
            with open(model_path, 'wb') as f:
                pickle.dump(self.l2r_ranker, f)

# Initialize function for app.py
def initialize():
    return EcoSearchEngine(max_docs=1000, use_l2r=True, multimodal=True)
