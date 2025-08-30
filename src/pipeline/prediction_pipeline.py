# prediction_pipeline.py - Prediction pipeline for Fake News Detection
import sys
import logging
from dataclasses import dataclass
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.exception import CustomException
from src.logger import logging
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

@dataclass
class PredictionPipelineConfig:
    def __post_init__(self):
        from src.utils import read_yaml
        config = read_yaml("config/urls_config.yaml")
        self.model_path: str = config["model_trainer"]["model_path"]
        self.preprocessor_path: str = config["data_transformation"]["preprocessed_object_path"]
        self.input_length: int = config["model_trainer"]["input_length"]

class PredictionPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()
        self.stop_words = set(stopwords.words("english"))
        self.tokenizer = None
        self.model = None

    def load_tokenizer(self):
        try:
            self.tokenizer = joblib.load(self.config.preprocessor_path)["tokenizer"]
            logging.info("Tokenizer loaded successfully")
        except Exception as e:
            raise CustomException(e, sys)

    def load_model(self):
        try:
            if not self.model:
                self.model = load_model(self.config.model_path)
                logging.info("Model loaded successfully")
        except Exception as e:
            raise CustomException(e, sys)

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

    def preprocess_text(self, text: str):
        try:
            if not self.tokenizer:
                self.load_tokenizer()
            cleaned_text = self.clean_text(text)
            sequences = self.tokenizer.texts_to_sequences([cleaned_text])
            padded_sequences = pad_sequences(sequences, maxlen=self.config.input_length, padding='post', truncating='post')
            return padded_sequences
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, text: str):
        try:
            processed_text = self.preprocess_text(text)
            self.load_model()
            prob_real = float(self.model.predict(processed_text)[0][0])
            prob_fake = 1 - prob_real
            label = int(prob_real > 0.3)  # Using 0.3 as threshold for real news
            # Return label, fake confidence %, real confidence %
            return label, prob_fake * 100, prob_real * 100
        except Exception as e:
            raise CustomException(e, sys)

    def predict_batch(self, texts: list):
        try:
            results = []
            for text in texts:
                results.append(self.predict(text))
            return results
        except Exception as e:
            raise CustomException(e, sys)
