"""
This file is to combine datasets and label eco-friendliness 
(and a lot of other preprocessing).

Date: Dec 1, 2024
Author: Sylvie Mei
"""

import gzip
import json
import os
import pickle
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from document_preprocessor import RegexTokenizer
import random
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'data/'  # TODO: Set this to the path to your data folder
CACHE_PATH = '__pycache__/'  # Set this to the path of the cache folder

BEAUTY_PATH = DATA_PATH + 'meta_All_Beauty.jsonl.gz'
FASHION_PATH = DATA_PATH + 'meta_Amazon_Fashion.jsonl.gz'
COMBINE_PATH = DATA_PATH + 'Beauty_and_Fashion.jsonl.gz'
STOPWORD_PATH = DATA_PATH + 'stopwords.txt'
MAIN_INDEX = 'main_index'
TITLE_INDEX = 'title_index'
N_DOC_NEEDED = 50
DOCID_TO_TITLE_PATH = CACHE_PATH + 'docid_to_title.pkl'
DOCID_TO_LINK_PATH = CACHE_PATH + 'docid_to_link.pkl'
DOCID_TO_IMAGE_PATH = CACHE_PATH + 'docid_to_image.pkl'
DOCID_TO_ASIN_PATH = CACHE_PATH + 'docid_to_asin.pkl'
DOCID_TO_DESC_PATH = CACHE_PATH + 'docid_to_desc.pkl'

DATASET_PATHS = [BEAUTY_PATH, FASHION_PATH]
KEYS_TO_KEEP = ["main_category", "title", "average_rating", 
                "rating_number", "price", "images", "details"]

# TODO: Modify positive and negative keywords
ecofriendly_keywords = ['sustainable', 'organic', 'biodegradable', 'recyclable', 
                        'compostable', 'recycled', 'non-toxic', 'renewable', 
                        'plant-based', 'vegan', 'low-impact', 'zero-waste', 
                        'green', 'cruelty-free', 'FSC-certified', 'carbon-neutral', 
                        'Energy Star', 'Fair Trade', 'eco-conscious', 
                        'climate-positive', 'upcycled', 'responsibly sourced', 
                        'energy-efficient', 'plastic-free', 'pesticide-free', 
                        'natural', 'ethical', 'eco-label', 'water-saving', 
                        'low-carbon', 'toxin-free', 'green-certified', 'eco-safe', 
                        'stainless steel', 'chemical-free', 'sulfate-free', 'paraben-free']
nonfriendly_keywords = ['non-recyclable', 'disposable', 'single-use']

class DatasetPreprocessor:
    def __init__(self, dataset_paths: list[str], combined_path: str, keys_to_keep: list[str]) -> None:
        self.dataset_paths = dataset_paths
        self.combined_path = combined_path
        self.keys_to_keep = keys_to_keep
        
    def combine_dataset(self) -> None:
        """
        Create combined dataset from input dataset list.
        """
        total_cnt = 0
        with gzip.open(self.combined_path, 'wt') as outfile:
            for input_path in self.dataset_paths:
                dataset_item_cnt = 0
                with gzip.open(input_path, 'r') as infile:
                    for line in infile:
                        data = json.loads(line)
                        if data['description'] == [] and data['features'] == []:
                            continue
                        dataset_item_cnt += 1
                        total_cnt += 1
                        filtered_data = {key:data[key] for key in self.keys_to_keep if key in data}
                        filtered_data['docid'] = total_cnt
                        filtered_data['description'] = " ".join(data['description'] + data['features'])
                        filtered_data['link'] = "https://www.amazon.com/dp/" + data['parent_asin']
                        
                        outfile.write(json.dumps(filtered_data) + '\n')
                    
                    print(f'Added {dataset_item_cnt} items from {input_path}.')
        
        print(f'Added {total_cnt} items in total to {self.combined_path}')

        return
    
    def filter_eco_keywords(self, eco_keywords: list[str], noneco_keywords: list[str], keyword_filtered_path = 'data/eco_keyword_labeled.jsonl.gz') -> None:
        """
        Label eco-fridnedliness based on keywords filtering.
        
        Args:
        eco_keywords
        """
        with gzip.open(keyword_filtered_path, 'wt') as outfile:
            with gzip.open(self.combined_path, 'rt') as file:
                for line in file:
                    data = json.loads(line)
                    description = data.get('description', '').lower()

                    eco_label = False
                    if any(eco_word in description for eco_word in eco_keywords):
                        if not any(noneco_word in description for noneco_word in noneco_keywords):
                            eco_label = True

                    data['eco_friendly'] = eco_label
                    outfile.write(json.dumps(data) + '\n')
    
    def fine_tune(self, pretrained_model: str = "distilbert-base-uncased-finetuned-sst-2-english", 
                  max_samples: int = 5000, num_epochs: int = 3, output_dir: str = "./fine_tuned_model", 
                  keyword_filtered_path = 'data/eco_keyword_labeled.jsonl.gz') -> None:
        """
        Fine-tune a pre-trained DistilBERT model using keyword-filtered labeled data.
        """
        with gzip.open(keyword_filtered_path, 'rt') as infile:
            lines = infile.readlines()

        labeled_data = []
        for line in lines:
            data = json.loads(line)
            if 'eco_friendly' not in data:
                raise ValueError("The dataset does not have eco_friendly labels. Please run filter_eco_keywords first.")
            
            label = 1 if data['eco_friendly'] else 0
            labeled_data.append({"text": data['description'], "label": label})

        labeled_data = random.sample(labeled_data, min(len(labeled_data), max_samples))
        train_data, val_data = train_test_split(labeled_data, test_size=0.2, random_state=650)
        
        tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
        def tokenize_data(data):
            encoding = tokenizer(data['text'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            data['input_ids'] = encoding['input_ids'].squeeze().tolist()  # Convert tensor to list
            data['attention_mask'] = encoding['attention_mask'].squeeze().tolist()  # Convert tensor to list
            return data

        train_data = [tokenize_data(data) for data in train_data]
        val_data = [tokenize_data(data) for data in val_data]
        
        train_dataset = Dataset.from_list(train_data).with_format("torch")
        val_dataset = Dataset.from_list(val_data).with_format("torch")

        model = DistilBertForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)

        
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=500,
            per_device_train_batch_size=16,
            num_train_epochs=num_epochs,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_total_limit=2,
            report_to="none",  # Disable reporting
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,  # Add validation dataset here
            tokenizer=tokenizer,
        )

        trainer.train()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned model saved to {output_dir}.")

    def filter_eco_sentiment(self, model_path: str = "./fine_tuned_model", 
                             keyword_filtered_path = 'data/eco_keyword_labeled.jsonl.gz', 
                             sentiment_filtered_path = 'data/sentiment_labeled.jsonl.gz') -> None:
        """
        Label eco-friendliness based on sentiment analysis using a fine-tuned model.
        """
        device="cuda" if torch.cuda.is_available() else "cpu"
        
        # Load fine-tuned model and tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

        # Update dataset with sentiment-based labels
        with gzip.open(keyword_filtered_path, 'rt') as infile:
            lines = infile.readlines()

        updated_data = []
        count = 0
        for line in tqdm(lines):
            data = json.loads(line)
            count += 1
            if count > 20000:
                break
            description = data.get('description', '')
            sentiment = sentiment_analyzer(description[:512])  # Truncate description to fit model input size
            data['eco_friendly'] = sentiment[0]['label'] == 'POSITIVE'
            # print(data['eco_friendly'])
            updated_data.append(data)

        with gzip.open(sentiment_filtered_path, 'wt') as outfile:
            for data in updated_data:
                outfile.write(json.dumps(data) + '\n')

        print(f"Sentiment-based eco-friendliness labeling completed and updated in {sentiment_filtered_path}")
    
    
def main():
    dataset_preprocessor = DatasetPreprocessor(DATASET_PATHS, COMBINE_PATH, KEYS_TO_KEEP)
    if not os.path.exists(COMBINE_PATH):
        dataset_preprocessor.combine_dataset()
    if not os.path.exists('data/eco_keyword_labeled.jsonl.gz'):
        dataset_preprocessor.filter_eco_keywords(eco_keywords=ecofriendly_keywords, noneco_keywords=nonfriendly_keywords)
    if not os.path.exists('./fine_tuned_model'):
        dataset_preprocessor.fine_tune()
    # if not os.path.exists('data/sentiment_labeled.jsonl.gz'):
    dataset_preprocessor.filter_eco_sentiment()

if __name__ == '__main__':
    main()
    