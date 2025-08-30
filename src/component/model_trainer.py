import os
import sys
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, ReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from src.exception import CustomException
from src.logger import logging
from src.utils import read_yaml
from src.component.data_transformation import DataTransformation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling1D
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
        self.dropout_rate = config["model_trainer"]["dropout_rate"]  # will use 0.7 like your example
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
        model = Sequential()

        # Embedding layer
        model.add(Embedding(
            input_dim=self.data_transformer.config.max_vocab_size,
            output_dim=self.config.output_dim,
            input_length=self.config.input_length
        ))
        model.layers[0].trainable = False  # freeze embeddings like your example
        # Replace Flatten with GlobalAveragePooling1D
        
        model.add(GlobalAveragePooling1D())

        

        # Hidden layers with ReLU + Dropout
        model.add(Dense(32,kernel_regularizer=l2(0.008)))
        model.add(ReLU())
        model.add(Dropout(self.config.dropout_rate))
        model.add(BatchNormalization())

        model.add(Dense(16,kernel_regularizer=l2(0.008)))
        model.add(ReLU())
        model.add(Dropout(self.config.dropout_rate))
        model.add(BatchNormalization())
        
        model.add(Dense(1, activation="sigmoid"))
        
        # Compile
        model.compile(
            optimizer=self.config.optimizer,
            loss=self.config.loss,
            metrics=self.config.metrics
        )

        logging.info("✅ ANN Model built and compiled successfully.")
        model.summary(print_fn=logging.info)
        return model

    def load_data(self):
        """Load datasets saved by DataTransformation (assumed as .npy)."""
        try:
            logging.info("Loading datasets...")
            data = {
                "X_train": np.load(self.config.train_data_path, allow_pickle=True),
                "y_train": np.load(self.config.train_labels_path, allow_pickle=True),
                "X_test": np.load(self.config.test_data_path, allow_pickle=True),
                "y_test": np.load(self.config.test_labels_path, allow_pickle=True),
                "X_val": np.load(self.config.val_data_path, allow_pickle=True),
                "y_val": np.load(self.config.val_labels_path, allow_pickle=True)
            }
            return data
        except Exception as e:
            raise CustomException(f"Error loading datasets: {e}", sys)

    def train_model(self):
        try:
            # Load data
            data = self.load_data()
            X_train, y_train = data["X_train"], data["y_train"]
            X_test, y_test = data["X_test"], data["y_test"]
            X_val, y_val = data["X_val"], data["y_val"]
            logging.info("Datasets loaded successfully.")
            logging.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
            logging.info(f"Train labels shape: {y_train.shape}, Val labels shape: {y_val.shape}, Test labels shape: {y_test.shape}")
            # Load preprocessor
            
            # Build model
            model = self.build_model()

            # Early stopping
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6 
            )
            # Train
            logging.info("🚀 Training started...")
            history = model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping,reduce_lr],
                verbose=1
            )
            # logging all parameters one by one
            logging.info(f"Model architecture: {model.summary()}")
            logging.info(f"Optimizer: {model.optimizer}")
            logging.info(f"Loss: {model.loss}")
            logging.info(f"Metrics: {model.metrics}")
            logging.info(f"Epochs: {self.config.epochs}")
            logging.info(f"Batch Size: {self.config.batch_size}")
            logging.info(f"Input Length: {self.config.input_length}")
            logging.info(f"Output Dim: {self.config.output_dim}")
            logging.info(f"Dropout Rate: {self.config.dropout_rate}")
            logging.info(f"Optimizer: {self.config.optimizer}")
            logging.info(f"Loss: {self.config.loss}")
            logging.info(f"Metrics: {self.config.metrics}")
            logging.info(f"Expected Accuracy: {self.config.expected_accuracy}")
            logging.info(f"Overfitting Threshold: {self.config.overfitting_threshold}")
            logging.info(f"Underfitting Threshold: {self.config.underfitting_threshold}")
            logging.info(f"Early Stopping Patience: {self.config.early_stopping_patience}")
            
            
            # logging every epoch in table
            history_df = pd.DataFrame(history.history)
            logging.info(f"\n{history_df}")
            logging.info("✅ Training completed.")
            

            # Evaluate after training
            train_metrics = model.evaluate(X_train, y_train, verbose=0)
            val_metrics = model.evaluate(X_val, y_val, verbose=0)
            test_metrics = model.evaluate(X_test, y_test, verbose=0)

            metrics_names = model.metrics_names
            train_results = dict(zip(metrics_names, train_metrics))
            val_results = dict(zip(metrics_names, val_metrics))
            test_results = dict(zip(metrics_names, test_metrics))

            logging.info(f"📊 Train Results: {train_results}")
            logging.info(f"📊 Val Results: {val_results}")
            logging.info(f"📊 Test Results: {test_results}")

            # Extract accuracy safely
            train_acc = train_results.get("accuracy", None)
            val_acc = val_results.get("accuracy", None)
            test_acc = test_results.get("accuracy", None)

            logging.info(f"Training Accuracy: {train_acc}")
            logging.info(f"Validation Accuracy: {val_acc}")
            logging.info(f"Test Accuracy: {test_acc}")
            

            # Overfitting / underfitting checks
            if (train_acc - val_acc) > self.config.overfitting_threshold:
                logging.warning("⚠️ Model is overfitting.")
                logging.info(f"overfitting by {train_acc - val_acc}")
            else:
                logging.info("✅ Model is not overfitting.")
                logging.info(f" diffrence score is {train_acc - val_acc}")
            if (val_acc - train_acc) > self.config.underfitting_threshold:
                logging.warning("⚠️ Model is underfitting.")
            if test_acc < self.config.expected_accuracy:
                logging.warning(f"⚠️ Model accuracy is below expected threshold by {self.config.expected_accuracy - test_acc}")

            # Save model
            os.makedirs(os.path.dirname(self.config.model_file_path), exist_ok=True)
            model.save(self.config.model_file_path)
            logging.info(f"✅ Model saved at {self.config.model_file_path}")

            

            return history

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model()
