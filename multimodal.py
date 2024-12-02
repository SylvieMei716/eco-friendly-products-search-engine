import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import requests
from io import BytesIO

# model = SentenceTransformer('clip-ViT-B-32')

# def get_image(url):
#     return Image.open(requests.get(url, stream=True).raw)

# doc1_img_emb = model.encode(get_image('https://m.media-amazon.com/images/I/31TgqAZ8kQL.jpg'))

# text_emb = model.encode(['plunger bars'])

# cos_scores = util.cos_sim(doc1_img_emb, text_emb)
# print(cos_scores)


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
            raise ValueError("Failed to load the image. Please check the URL.")
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

    def compute_similarity(self, image_url: str, texts: list[str]):
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
        cos_scores = util.cos_sim(image_emb, text_emb)
        return cos_scores.tolist()

# Example usage
if __name__ == "__main__":
    # Initialize the multimodal search
    search = MultimodalSearch(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Example query and image URLs
    query = "eco-friendly beauty product"
    image_urls = [
        "https://m.media-amazon.com/images/I/71i77AuI9xL._SL1500_.jpg",
        "https://m.media-amazon.com/images/I/41qfjSfqNyL.jpg",
        "https://m.media-amazon.com/images/I/41w2yznfuZL.jpg"
    ]  # Replace with your actual image URLs
    
    # Rank the images by their similarity to the query
    results = search.rank_images(query, image_urls)
    
    # Print the ranked results
    for image_url, similarity in results:
        print(f"Image URL: {image_url}, Similarity: {similarity:.4f}")