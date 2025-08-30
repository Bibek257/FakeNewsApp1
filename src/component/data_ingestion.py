import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.utils import read_yaml
from dataclasses import dataclass
import sys


@dataclass
class DataIngestion_config:
    def __post_init__(self):
        config = read_yaml("config/urls_config.yaml")
        self.fake_news_data_path = config['data_ingestion']['fake_news_data_path']
        self.real_news_data_path = config['data_ingestion']['real_news_data_path']
        self.raw_data_dir = config['data_ingestion']['raw_data_dir']
        self.train_data_path = config['data_ingestion']['train_data_path']
        self.test_data_path = config['data_ingestion']['test_data_path']
        self.val_data_path = config['data_ingestion']['val_data_path']

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestion_config()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df_fake = pd.read_csv(self.ingestion_config.fake_news_data_path)
            df_real = pd.read_csv(self.ingestion_config.real_news_data_path)
            df_fake["class"] = 0
            df_real["class"] = 1
            df = pd.concat([df_fake, df_real], axis=0)
            logging.info("Read the dataset as dataframe")
            df = df.sample(frac=1).reset_index(drop=True)
            df=df.drop(columns=['title', 'subject', 'date'], axis=1)
            # ensiuring the directory is present
            os.makedirs(self.ingestion_config.raw_data_dir, exist_ok=True)
            # saving the raw data
            df.to_csv(self.ingestion_config.raw_data_dir+"/raw_data.csv", index=False, header=True)
            logging.info("Raw data is saved")
            # dropping duplicates
            df = df.drop_duplicates().reset_index(drop=True)
            # saving the train and test data
            train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)
            train_set,val_set = train_test_split(train_set, test_size=0.15, random_state=42)
             # 0.25 x 0.8 = 0.2
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            logging.info("Train, validation and Test data is saved")
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.val_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, val_data = obj.initiate_data_ingestion()
    
    
        
 