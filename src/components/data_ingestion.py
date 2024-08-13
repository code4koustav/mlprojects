import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_tranformation import DataTransformation
from src.components.data_tranformation import DataTransformationConfig


# Define a dataclass for storing data ingestion configurations
@dataclass
class DataIngestionConfig:
    # Paths for saving train, test, and raw data
    train_data_path = os.path.join('artifacts', "train.csv")
    test_data_path = os.path.join('artifacts', "test.csv")
    raw_data_path = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        # Initialize the ingestion configuration
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered into data ingestion component')  # Log entry into the data ingestion process

        try:
            # Read the dataset from a specified path
            df = pd.read_csv('notebooks/data/stud.csv')
            logging.info('Read the dataset')  # Log successful dataset read

            # Create necessary directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # Save the raw dataset to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")  # Log initiation of train-test split
            # Split the dataset into training and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test sets to the specified paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")  # Log completion of data ingestion

            # Return paths of the saved train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            # Handle exceptions by raising a custom exception with logging
            raise CustomException(e, sys)

# Main execution block
if __name__ == "__main__":
    obj = DataIngestion()  # Create an instance of DataIngestion
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

