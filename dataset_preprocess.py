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
import random

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
EDGELIST_PATH = DATA_PATH + 'edgelist.csv.gz'
NETWORK_STATS_PATH = DATA_PATH + 'network_stats.csv'
DOCID_TO_DESC_PATH = DATA_PATH + 'docid_to_desc.pkl'

DATASET_PATHS = [BEAUTY_PATH, FASHION_PATH]
KEYS_TO_KEEP = ["main_category", "title", "average_rating", 
                "rating_number", "price", "images", "details"]

ecofriendly_keywords = ['sustainable', 'organic', 'biodegradable', 'recyclable', 
                        'compostable', 'recycled', 'non-toxic', 'renewable', 
                        'plant-based', 'vegan', 'low-impact', 'zero-waste', 
                        'green', 'cruelty-free', 'FSC-certified', 'carbon-neutral', 
                        'Energy Star', 'Fair Trade', 'eco-conscious', 
                        'climate-positive', 'upcycled', 'responsibly sourced', 
                        'energy-efficient', 'plastic-free', 'pesticide-free', 
                        'natural', 'ethical', 'eco-label', 'water-saving', 
                        'low-carbon', 'toxin-free', 'green-certified', 'eco-safe']
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
        with gzip.open(self.combined_path, 'w') as outfile:
            for input_path in self.dataset_paths:
                dataset_item_cnt = 0
                with gzip.open(input_path, 'rt') as infile:
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
    
    def filter_eco_keywords(self, eco_keywords: list[str], noneco_keywords: list[str]) -> None:
        """
        Label eco-fridnedliness based on keywords filtering.
        
        Args:
        eco_keywords
        """
        with open(self.combined_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                description = data.get('description', '').lower()

                eco_label = False
                if any(eco_word in description for eco_word in eco_keywords):
                    if not any(noneco_word in description for noneco_word in noneco_keywords):
                        eco_label = True

                data['eco_friendly'] = eco_label
                file.write(json.dumps(data) + '\n')
        pass
    
    def fine_tune(self, pretrained_model: str = "distilbert-base-uncased-finetuned-sst-2-english", 
                  max_samples: int = 5000, num_epochs: int = 3, output_dir: str = "./fine_tuned_model") -> None:
        """
        Fine-tune a pre-trained DistilBERT model using keyword-filtered labeled data.
        """
        # Step 1: Load keyword-labeled data
        with open(self.combined_path, 'r') as infile:
            lines = infile.readlines()

        labeled_data = []
        for line in lines:
            data = json.loads(line)
            if 'eco_friendly' not in data:
                raise ValueError("The dataset does not have eco_friendly labels. Please run filter_eco_keywords first.")
            
            label = 1 if data['eco_friendly'] else 0
            labeled_data.append({"text": data['description'], "label": label})

        # Randomly sample max_samples rows
        labeled_data = random.sample(labeled_data, min(len(labeled_data), max_samples))
        
        # Step 2: Prepare dataset for training
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        dataset = Dataset.from_list(labeled_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        train_dataset = tokenized_dataset.with_format("torch")

        # Step 3: Initialize tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
        model = DistilBertForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)

        # Step 4: Define training arguments
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
            warmup=600,
            learning_rate = 1e-5,
            max_seq_length = 128,
        )

        # Step 5: Define Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

        # Step 6: Train the model
        trainer.train()

        # Step 7: Save the fine-tuned model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned model saved to {output_dir}.")

    def filter_eco_sentiment(self, model_path: str = "./fine_tuned_model") -> None:
        """
        Label eco-friendliness based on sentiment analysis using a fine-tuned model.
        """
        # Load fine-tuned model and tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        # Update dataset with sentiment-based labels
        with open(self.combined_path, 'r') as infile:
            lines = infile.readlines()

        updated_data = []
        for line in lines:
            data = json.loads(line)
            description = data.get('description', '')
            sentiment = sentiment_analyzer(description[:512])  # Truncate description to fit model input size
            data['eco_friendly'] = sentiment[0]['label'] == 'LABEL_1'
            updated_data.append(data)

        with open(self.combined_path, 'w') as outfile:
            for data in updated_data:
                outfile.write(json.dumps(data) + '\n')

        print(f"Sentiment-based eco-friendliness labeling completed and updated in {self.combined_path}")
    
    
def main():
    dataset_preprocessor = DatasetPreprocessor(DATASET_PATHS, COMBINE_PATH)
    dataset_preprocessor.combine_dataset()
    dataset_preprocessor.filter_eco_keywords(eco_keywords=ecofriendly_keywords, noneco_keywords=nonfriendly_keywords)
    dataset_preprocessor.fine_tune()
    dataset_preprocessor.filter_eco_sentiment()

if __name__ == '__main__':
    main()
    