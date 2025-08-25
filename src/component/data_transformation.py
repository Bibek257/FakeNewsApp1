from gensim.models import KeyedVectors
vectors = KeyedVectors.load("word2vec_vectors.kv")
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys
import re
import string
import os
import joblib
from dataclasses import dataclass
from src.utils import read_yaml

@dataclass
class DataTransformationConfig:
    def __post_init__(self):
        config = read_yaml("config/urls_config.yaml")
        self.preprocessed_object_file_path = config["data_transformation"]["preprocessed_object_file_path"]
        self.valid_train_data_path = config["data_validation"]["valid_train_data_path"]
        self.valid_test_data_path = config["data_validation"]["valid_test_data_path"]
        self.valid_val_data_path = config["data_validation"]["valid_val_data_path"]
        self.transformed_train_path = config["data_transformation"]["transformed_train_path"]
        self.transformed_test_path = config["data_transformation"]["transformed_test_path"]
        self.transformed_val_path = config["data_transformation"]["transformed_val_path"]
        self.word2vec_model_path = config["word2vec"]["word2vec_model_path"]
        self.embeded_train_path = config["data_transformation"]["embeded_train_path"]
        self.embeded_test_path = config["data_transformation"]["embeded_test_path"]
        self.embeded_val_path = config["data_transformation"]["embeded_val_path"]
        self.max_sequence_length = config["data_transformation"]["max_sequence_length"]
        
# class to clan, tookrenize, remove stopwords and embed the text data using word2vec and save it in diffrent files for train, test and val dataset
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.stop_words = set(stopwords.words('english'))
        self.word2vec_model = KeyedVectors.load(self.data_transformation_config.word2vec_model_path)
        logging.info("Word2Vec model loaded successfully.")

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\w*\d\w*', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        return " ".join(tokens)

    def embed_and_pad(self, df: pd.DataFrame, save_path: str):
        embeddings = []
        for text in df['text']:
            tokens = word_tokenize(text)
            vectors = [self.word2vec_model[word] for word in tokens if word in self.word2vec_model]
            if vectors:
                avg_vector = np.mean(vectors, axis=0)
            else:
                avg_vector = np.zeros(self.word2vec_model.vector_size)
            embeddings.append(avg_vector)
        embeddings = np.array(embeddings)
        np.save(save_path, embeddings)  # this creates .npy file
        return embeddings

    def initiate_data_transformation(self):
        try:
            # Load validated datasets
            train_df = pd.read_csv(self.data_transformation_config.valid_train_data_path)
            test_df = pd.read_csv(self.data_transformation_config.valid_test_data_path)
            val_df = pd.read_csv(self.data_transformation_config.valid_val_data_path)

            # Clean text
            for df in [train_df, test_df, val_df]:
                df['text'] = df['text'].apply(self.clean_text)

            # Save cleaned text
            train_df.to_csv(self.data_transformation_config.transformed_train_path, index=False)
            test_df.to_csv(self.data_transformation_config.transformed_test_path, index=False)
            val_df.to_csv(self.data_transformation_config.transformed_val_path, index=False)

            # Embed and save
            self.embed_and_pad(train_df, self.data_transformation_config.embeded_train_path)
            self.embed_and_pad(test_df, self.data_transformation_config.embeded_test_path)
            self.embed_and_pad(val_df, self.data_transformation_config.embeded_val_path)

            # Save preprocessor (just stopwords + clean_text config)
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessed_object_file_path), exist_ok=True)
            import joblib
            joblib.dump({"stop_words": self.stop_words}, self.data_transformation_config.preprocessed_object_file_path)

        except Exception as e:
            raise CustomException(e, sys)
