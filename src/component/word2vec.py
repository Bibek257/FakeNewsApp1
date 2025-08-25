# this file is for word2vec model downloading and saving
import nltk
import gensim.downloader as api
from gensim.models import Word2Vec
from src.utils import read_yaml
from dataclasses import dataclass
import os
import sys
from src.exception import CustomException
from src.logger import logging

@dataclass
class Word2VecModelConfig:
    def __init__(self, config):
        config = read_yaml("config/urls_config.yaml")
        self.model_path = config['word2vec']['word2vec_model_path']
        
class Word2VecModel:
    """Class to handle downloading and saving the Word2Vec model.
    
    Attributes:
        model_config (Word2VecModelConfig): Configuration for the Word2Vec model.
    
    Methods:
        download_and_save_model(): Downloads and saves the Word2Vec model.
        
        """

    def __init__(self):
        self.model_config = Word2VecModelConfig(config={})
    
    def download_and_save_model(self):
        try:
            # Download the pre-trained Word2Vec model
            # check if model is already downloaded
            if os.path.exists(self.model_config.model_path):
                logging.info(f"Word2Vec model already exists at {self.model_config.model_path}. Skipping download.")
                return
            else:
                logging.info("Downloading Word2Vec model...")
                model = api.load("word2vec-google-news-300")
                logging.info("Word2Vec model downloaded successfully.")
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.model_config.model_path), exist_ok=True)
            
            # Save the model
            logging.info(f"Saving Word2Vec model to {self.model_config.model_path}...")
            model.save(self.model_config.model_path)
            logging.info("Word2Vec model saved successfully.")
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    w2v_model = Word2VecModel()
    w2v_model.download_and_save_model()
