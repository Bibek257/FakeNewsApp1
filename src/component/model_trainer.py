import os
import sys
import numpy as np
import joblib
from dataclasses import dataclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml
from src.component.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    def __post_init__(self):
        config = read_yaml("config/urls_config.yaml")
        # Paths
        self.model_file_path = config["model_trainer"]["model_path"]
        self.train_data_path = config["data_transformation"]["embeded_train_path"]
        self.train_labels_path = config["data_transformation"]["embeded_y_train_path"]
        self.test_data_path = config["data_transformation"]["embeded_test_path"]
        self.test_labels_path = config["data_transformation"]["embeded_y_test_path"]
        self.val_data_path = config["data_transformation"]["embeded_val_path"]
        self.val_labels_path = config["data_transformation"]["embeded_y_val_path"]

        # Hyperparams
        self.epochs = config["model_trainer"]["epochs"]
        self.batch_size = config["model_trainer"]["batch_size"]
        self.input_length = config["model_trainer"]["input_length"]
        self.output_dim = config["model_trainer"]["output_dim"]
        self.dropout_rate = config["model_trainer"]["dropout_rate"]
        self.optimizer = config["model_trainer"]["optimizer"]
        self.loss = config["model_trainer"]["loss"]
        self.metrics = config["model_trainer"]["metrics"]
        self.expected_accuracy = config["model_trainer"]["expected_accuracy"]
        self.overfitting_threshold = config["model_trainer"]["overfitting_threshold"]
        self.underfitting_threshold = config["model_trainer"]["underfitting_threshold"]
        self.early_stopping_patience = config["model_trainer"]["early_stopping_patience"]
        self.preprocessor_path = config["data_transformation"]["preprocessed_object_path"]


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.data_transformer = DataTransformation()

    def build_model(self):
        model = Sequential([
            Embedding(input_dim=self.data_transformer.config.max_vocab_size,
                      output_dim=self.config.output_dim,
                      input_length=self.config.input_length),
            GlobalAveragePooling1D(),
            Dropout(self.config.dropout_rate),
            Dense(128, activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(64, activation='relu'),
            Dropout(self.config.dropout_rate),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=self.config.optimizer,
                      loss=self.config.loss,
                      metrics=self.config.metrics,
                      )

        logging.info("✅ Model built and compiled successfully.")
        return model

    def load_data(self):
        """Load datasets saved by DataTransformation (assumed as .npy)."""
        try:
            logging.info("Loading datasets...")
            X_train = np.load(self.config.train_data_path, allow_pickle=True)
            y_train = np.load(self.config.train_labels_path, allow_pickle=True)
            X_test = np.load(self.config.test_data_path, allow_pickle=True)
            y_test = np.load(self.config.test_labels_path, allow_pickle=True)
            X_val = np.load(self.config.val_data_path, allow_pickle=True)
            y_val = np.load(self.config.val_labels_path, allow_pickle=True)
            logging.info("✅ Datasets loaded successfully.")
            return X_train, y_train, X_test, y_test, X_val, y_val
        except Exception as e:
            raise CustomException(f"Error loading datasets: {e}", sys)

    def train_model(self):
        try:
            X_train, y_train, X_test, y_test, X_val, y_val = self.load_data()

            # Build model
            model = self.build_model()

            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            )

            # Train
            logging.info("🚀 Training started...")
            history = model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )

            # Evaluate
            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

            logging.info(f"📊 Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

            # Overfitting / underfitting checks
            if (train_acc - val_acc) > self.config.overfitting_threshold:
                raise CustomException("⚠️ Model is overfitting.")
            if (val_acc - train_acc) > self.config.underfitting_threshold:
                raise CustomException("⚠️ Model is underfitting.")
            if test_acc < self.config.expected_accuracy:
                raise CustomException("⚠️ Model accuracy is below expected threshold.")

            # Save model
            os.makedirs(os.path.dirname(self.config.model_file_path), exist_ok=True)
            model.save(self.config.model_file_path)
            logging.info(f"✅ Model saved at {self.config.model_file_path}")

            # Save preprocessor
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            joblib.dump(self.data_transformer, self.config.preprocessor_path)
            logging.info(f"✅ Preprocessor saved at {self.config.preprocessor_path}")

            return history

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model()
