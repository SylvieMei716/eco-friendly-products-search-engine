"""
This file is to implement CLIP method to calculate the similarity 
between query and item image.

Date: Dec 2, 2024
Author: Sylvie Mei
"""

import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import os
import json
import pickle
from tqdm import tqdm
from PIL import UnidentifiedImageError

class MultimodalSearch:
    def __init__(self, model_name: str = 'clip-ViT-B-32', embeddings_file: str='./__pycache__/image_embeddings.pkl') -> None:
        """
        Initialize the MultimodalSearch class with a pre-trained model.
        
        Args:
        - model_name (str): The name of the pre-trained model to use for encoding.
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings_file = embeddings_file
        self.image_embeddings = self.load_embeddings()

    def get_image(self, url: str) -> Image.Image:
        """
        Fetch and load an image from a URL.
        
        Args:
        - url (str): The URL of the image.
        
        Returns:
        - PIL.Image.Image: The loaded image.
        """
        if url == "":
            return None
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            return Image.open(response.raw).convert("RGB")  # Ensure image is in RGB mode
        except Exception as e:
            print(f"Error fetching image: {e}")
            return None

    def encode_image(self, image_url: str):
        """
        Encode an image using the pre-trained model.
        
        Args:
        - image_url (str): The URL of the image to encode.
        
        Returns:
        - ndarray: The image embedding.
        """
        if image_url in self.image_embeddings:
            return np.array(self.image_embeddings[image_url])
        
        image = self.get_image(image_url)
        if image is None:
            # raise ValueError(f"Failed to load the image. Please check the URL {image_url}")
            return np.array([0.0])
        return self.model.encode(image)

    def encode_text(self, texts: list[str]):
        """
        Encode a list of texts using the pre-trained model.
        
        Args:
        - texts (list[str]): A list of strings to encode.
        
        Returns:
        - ndarray: The text embeddings.
        """
        return self.model.encode(texts)

    def compute_similarity(self, image_url: str, texts: list[str]) -> list[float]:
        """
        Compute the cosine similarity between an image and a list of text descriptions.
        
        Args:
        - image_url (str): The URL of the image.
        - texts (list[str]): A list of text descriptions.
        
        Returns:
        - list[float]: The similarity scores for each text.
        """
        image_emb = self.encode_image(image_url)
        text_emb = self.encode_text(texts)
        if np.array_equal(image_emb, np.array([0.0])):
        # If the image embedding is invalid, return a list of zeros for each text description
            image_emb = np.zeros_like(text_emb)
            # print(image_emb)
            # print(len(text_emb))
        
        cos_scores = util.cos_sim(image_emb, text_emb)
        return cos_scores.tolist()

    def save_embeddings(self) -> None:
        """
        Save the current image embeddings to a file.
        """
        pickle.dump(self.image_embeddings, open(self.embeddings_file, 'wb'))

    def load_embeddings(self) -> None:
        """
        Load image embeddings from a file.
        
        Returns:
        - dict: A dictionary of image URLs and their embeddings.
        """
        if os.path.exists(self.embeddings_file):
            # with open(self.embeddings_file, 'r') as f:
            return pickle.load(open(self.embeddings_file, 'rb'))
        return {}
    
    def precompute_embeddings(self, image_urls: list[str]) -> None:
        """
        Precompute embeddings for a list of image URLs and save them to the embeddings file every 1000 steps.
        
        Args:
        - image_urls (list[str]): A list of image URLs to encode.
        """
        processed_urls = set(self.image_embeddings.keys())
        print(f'processed images number: {len(processed_urls)}')

        remaining_urls = [url for url in image_urls if url not in processed_urls]

        for i, url in enumerate(tqdm(remaining_urls, desc="Encoding images")):
            self.image_embeddings[url] = self.encode_image(url)
            if (i + 1) % 1000 == 0 or (i + 1) == len(remaining_urls):
                self.save_embeddings()
                print(f"Checkpoint: Saved embeddings at step {i + 1}/{len(remaining_urls)}")
        
def main():
    CACHE_PATH = './__pycache__/'
    DOCID_TO_IMAGE_PATH = CACHE_PATH + 'docid_to_image.pkl'
    docid_to_image = pickle.load(open(DOCID_TO_IMAGE_PATH, 'rb'))
    image_urls = [url for docid, url in docid_to_image.items()]
    print(f'total images number: {len(image_urls)}')
    clip = MultimodalSearch()
    clip.precompute_embeddings(image_urls)


if __name__ == '__main__':
    main()