import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import joblib
import re
import string
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

@dataclass
class DataTransformationConfig:
    def __post_init__(self):
        config = read_yaml("config/urls_config.yaml")
        self.preprocessed_object_file_path = "transformer/preprocessor.pkl"
        self.valid_train_data_path = config["data_validation"]["valid_train_data_path"]
        self.valid_test_data_path = config["data_validation"]["valid_test_data_path"]
        self.valid_val_data_path = config["data_validation"]["valid_val_data_path"]
        self.word2vec_model_path = config["word2vec"]["word2vec_model_path"]
        self.embeded_train_path = config["data_transformation"]["embeded_train_path"]
        self.embeded_y_train_path = config["data_transformation"]["embeded_y_train_path"]
        self.embeded_test_path = config["data_transformation"]["embeded_test_path"]
        self.embeded_y_test_path = config["data_transformation"]["embeded_y_test_path"]
        self.embeded_val_path = config["data_transformation"]["embeded_val_path"]
        self.embeded_y_val_path = config["data_transformation"]["embeded_y_val_path"]
        self.max_sequence_length = config["data_transformation"]["max_sequence_length"]
        self.max_vocab_size = config["data_transformation"].get("max_vocab_size", 50000)
        self.embedding_dim = config["data_transformation"].get("embedding_dim", 300)
        self.padding_type = config["data_transformation"].get("padding_type", "post")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.stop_words = set(stopwords.words("english"))

        # Load Word2Vec
        self.word2vec_model = KeyedVectors.load(self.config.word2vec_model_path, mmap='r')
        logging.info("Word2Vec model loaded successfully.")

        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.config.max_vocab_size, oov_token="<OOV>")

    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>+", "", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\w*\d\w*", "", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        return " ".join(tokens)

    def fit_tokenizer(self, texts):
        self.tokenizer.fit_on_texts(texts)
        logging.info(f"Tokenizer fitted. Vocab size: {len(self.tokenizer.word_index)}")

    def texts_to_padded_sequences(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.config.max_sequence_length, padding='post', truncating='post')
        return padded

    def create_embedding_matrix(self):
        word_index = self.tokenizer.word_index
        vocab_size = min(self.config.max_vocab_size, len(word_index) + 1)
        embedding_matrix = np.zeros((vocab_size, self.config.embedding_dim))

        for word, i in word_index.items():
            if i >= self.config.max_vocab_size:
                continue
            if word in self.word2vec_model:
                embedding_matrix[i] = self.word2vec_model[word]

        return embedding_matrix

    def initiate_data_transformation(self):
        try:
            logging.info("Data transformation started...")

            # Load datasets
            train_data = pd.read_csv(self.config.valid_train_data_path)
            test_data = pd.read_csv(self.config.valid_test_data_path)
            val_data = pd.read_csv(self.config.valid_val_data_path)
            logging.info("Datasets loaded successfully.")

            # Clean text
            for df in [train_data, test_data, val_data]:
                df["text"] = df["text"].apply(self.clean_text)
            logging.info("Text cleaning completed.")

            # Fit tokenizer on training data
            self.fit_tokenizer(train_data["text"])

            # Convert texts to padded sequences
            X_train = self.texts_to_padded_sequences(train_data["text"])
            X_test = self.texts_to_padded_sequences(test_data["text"])
            X_val = self.texts_to_padded_sequences(val_data["text"])
            logging.info("Text to padded sequences conversion completed.")

            y_train = train_data["class"].values
            y_test = test_data["class"].values
            y_val = val_data["class"].values
            logging.info("Labels extracted.")

            # Ensure transformer folder exists
            os.makedirs("artifacts/data_transformation", exist_ok=True)

            # Save padded sequences
            np.save(self.config.embeded_train_path, (X_train))
            np.save(self.config.embeded_y_train_path, (y_train))
            np.save(self.config.embeded_test_path, (X_test))
            np.save(self.config.embeded_y_test_path, (y_test))
            np.save(self.config.embeded_val_path, (X_val))
            np.save(self.config.embeded_y_val_path, (y_val))
            logging.info("Padded sequences saved successfully.")

            # Save tokenizer and preprocessor info
            preprocessor_obj = {
                "tokenizer": self.tokenizer,
                "max_sequence_length": self.config.max_sequence_length,
                "embedding_matrix": self.create_embedding_matrix()
            }
            joblib.dump(preprocessor_obj, self.config.preprocessed_object_file_path)
            logging.info("Preprocessor.pkl saved successfully.")

            logging.info("Data transformation completed successfully.")
            return X_train, y_train, X_test, y_test, X_val, y_val

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataTransformation()
    obj.initiate_data_transformation()
