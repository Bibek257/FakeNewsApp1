# data validation
from src.utils import read_yaml
from dataclasses import dataclass
import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import numpy as np


@dataclass
class DataValidationConfig:
    def __post_init__(self):
        config = read_yaml("config/urls_config.yaml")
        self.valid_data_path = config["data_validation"]["valid_data_path"]
        self.train_data_path = config["data_ingestion"]["train_data_path"]
        self.test_data_path = config["data_ingestion"]["test_data_path"]
        self.val_data_path = config["data_ingestion"]["val_data_path"]
        self.train_valid_data_path=config['data_validation']['valid_train_data_path']
        self.test_valid_data_path=config['data_validation']['valid_test_data_path']
        self.val_valid_data_path=config['data_validation']['valid_val_data_path']
        self.smallest_acceptable_size=config['data_validation']['smallest_acceptable_size']
        self.largest_acceptable_size=config['data_validation']['largest_acceptable_size']
    
class DataValidation:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()
    
        
    def validate_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Validate dataframe by dropping rows with missing values.
            Logs number of rows dropped.
        """
        try:
            n_missing_before = df.isnull().sum().sum()
            df = df.dropna().reset_index(drop=True)
            n_missing_after = df.isnull().sum().sum()
            n_dropped = n_missing_before - n_missing_after

            if n_dropped > 0:
                logging.info(f"total missing values before: {n_missing_before}")
                logging.info(f"Dropped {n_dropped} rows with missing values")
                logging.info(f"total missing values after: {n_missing_after}")

            return df

        except Exception as e:
            raise CustomException(e, sys)   
        
    def check_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Validate dataframe by dropping duplicate rows.
            Logs number of rows dropped.
        """
        try:
            n_duplicates_before = df.duplicated().sum()
            df = df.drop_duplicates().reset_index(drop=True)
            n_duplicates_after = df.duplicated().sum()
            n_dropped = n_duplicates_before - n_duplicates_after

            if n_dropped > 0:
                logging.info(f"total duplicates before: {n_duplicates_before}")
                logging.info(f"Dropped {n_dropped} duplicate rows")
                logging.info(f"total duplicates after: {n_duplicates_after}")

            return df

        except Exception as e:
            raise CustomException(e, sys)   
    
    def validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            valid_labels = [0, 1]
            invalid_rows = df[~df['class'].isin(valid_labels)]
            n_invalid = invalid_rows.shape[0]

            if n_invalid > 0:
                logging.warning(f"Dropped {n_invalid} rows with invalid labels")
                df = df[df['class'].isin(valid_labels)]

            return df
        except Exception as e:
            raise CustomException(e, sys)


# method to initiate the validation iin all training, testing and validation data
    def initiate_data_validation(self):
        try:
            # Load datasets
            train_df = pd.read_csv(self.data_validation_config.train_data_path)
            test_df = pd.read_csv(self.data_validation_config.test_data_path)
            val_df = pd.read_csv(self.data_validation_config.val_data_path)

            # Validate datasets
            train_df = self.validate_missing_values(train_df)
            train_df = self.check_duplicates(train_df)
            
            train_df = self.validate_labels(train_df)

            test_df = self.validate_missing_values(test_df)
            test_df = self.check_duplicates(test_df)
           
            test_df = self.validate_labels(test_df)

            val_df = self.validate_missing_values(val_df)
            val_df = self.check_duplicates(val_df)
            
            val_df = self.validate_labels(val_df)

            # Save valid and invalid data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_data_path), exist_ok=True)
            
            train_df.to_csv(self.data_validation_config.train_valid_data_path, index=False)
            logging.info(f"Valid train data saved to {self.data_validation_config.train_valid_data_path}")
            test_df.to_csv(self.data_validation_config.test_valid_data_path, index=False)
            logging.info(f"Valid test data saved to {self.data_validation_config.test_valid_data_path}")
            val_df.to_csv(self.data_validation_config.val_valid_data_path, index=False)
            logging.info(f"Valid val data saved to {self.data_validation_config.val_valid_data_path}")
            logging.info(f"Valid data saved to {self.data_validation_config.valid_data_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataValidation()
    obj.initiate_data_validation()
    