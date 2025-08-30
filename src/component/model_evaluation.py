# This file consists of model evaluation functions and report generation
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import sys
from src.utils import read_yaml
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer, ModelTrainerConfig
from tensorflow.keras.models import load_model

@dataclass
class ModelEvaluationConfig:
    def __post_init__(self):
        config = read_yaml("config/urls_config.yaml")
        self.test_data_path = config["data_transformation"]["embeded_test_path"]
        self.test_labels_path = config["data_transformation"]["embeded_y_test_path"]
        self.model_path = config["model_trainer"]["model_path"]
        self.report_path = config["model_evaluation"]["model_evaluation_file_path"]
        self.expected_accuracy = config["model_trainer"]["expected_accuracy"]
        self.overfitting_threshold = config["model_trainer"]["overfitting_threshold"]
        self.underfitting_threshold = config["model_trainer"]["underfitting_threshold"]
        self.preprocessor_path = config["data_transformation"]["preprocessed_object_path"]
class ModelEvaluation:
    def __init__(self):
        self.model_eval_config = ModelEvaluationConfig()
        self.data_transformer = DataTransformation()
        self.model_trainer = ModelTrainer()

    def evaluate_model(self):
        logging.info("Model Evaluation started")
        try:
            # Load test data
            X_test = np.load(self.model_eval_config.test_data_path)
            y_test = np.load(self.model_eval_config.test_labels_path)

            # Load model
            model = load_model(self.model_eval_config.model_path)
            logging.info("Model loaded successfully")

            # Make predictions
            y_pred_prob = model.predict(X_test)
            # changing threshold to 0.6 from 0.5
            y_pred = (y_pred_prob > 0.6).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Generate classification report
            class_report = classification_report(y_test, y_pred)

            # Save report
            os.makedirs(os.path.dirname(self.model_eval_config.report_path), exist_ok=True)
            with open(self.model_eval_config.report_path, "w") as report_file:
                report_file.write(f"Accuracy: {accuracy}\n")
                report_file.write(f"Precision: {precision}\n")
                report_file.write(f"Recall: {recall}\n")
                report_file.write(f"F1 Score: {f1}\n\n")
                report_file.write("Classification Report:\n")
                report_file.write(class_report)

            logging.info("Model evaluation report generated successfully")
            

            logging.info("Model Evaluation completed")
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.evaluate_model()
        