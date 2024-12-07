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

class MultimodalSearch:
    def __init__(self, model_name: str = 'clip-ViT-B-32') -> None:
        """
        Initialize the MultimodalSearch class with a pre-trained model.
        
        Args:
        - model_name (str): The name of the pre-trained model to use for encoding.
        """
        self.model = SentenceTransformer(model_name)

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
            return Image.open(response.raw)
        except requests.exceptions.RequestException as e:
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
            print(image_emb)
            print(len(text_emb))
        
        cos_scores = util.cos_sim(image_emb, text_emb)
        return cos_scores.tolist()
