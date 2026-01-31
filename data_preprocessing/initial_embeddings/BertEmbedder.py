# Import necessary libraries
import random
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Define a class BertEmbedder for text embedding using BERT model
class BertEmbedder:
    def __init__(self, pretrained_model_name_or_path):
        # Set a fixed random seed for reproducibility
        random_seed = 42
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Enable CUDA if available
        if torch.cuda.is_available():
            print("\n**** CUDA is available. Using GPU! ****\n")
            torch.cuda.manual_seed_all(random_seed)

        # Initialize the tokenizer and model with the specified pretrained BERT model
        # self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        # self.model = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        # Move model to GPU if available
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Method to update tokenizer and model if a different pretrained model name is needed
    def set_pretrained_model_name(self, pretrained_model_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model = BertModel.from_pretrained(pretrained_model_name)
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Method to generate embedding for a given text entity
    def embed_entity(self, entity):
        # Prepare text inputs for the model
        encoding = self.tokenizer.batch_encode_plus(
            [entity],                 # List of input texts
            padding=True,             # Pad to the maximum sequence length
            truncation=True,          # Truncate to the maximum sequence length if necessary
            return_tensors='pt',      # Return PyTorch tensors
            add_special_tokens=True   # Add special tokens (CLS, SEP)
        )

        # Extract input_ids and attention masks
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Compute embeddings without updating model parameters
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.model.device), attention_mask=attention_mask.to(self.model.device))
            word_embeddings = outputs.last_hidden_state  # Extract embeddings

        # Calculate the mean of all word embeddings to represent the entity
        entity_embedding = word_embeddings.mean(dim=1)
        return entity_embedding
