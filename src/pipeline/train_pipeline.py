import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
#from src.components.data_transformation import DataTransformation
from src.components.data_retriver import ModelTrainer,DataTransformation

class TrainPipeline:
    def __init__(self):
        self.train_data_path = None
        self.test_data_path = None
        self.train_arr = None
        self.test_arr = None
        self.preprocessor_path = None
        self.r2_score = None

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting Data Ingestion")
            data_ingestion = DataIngestion()
            self.train_data_path, self.test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed. Train data path: {self.train_data_path}, Test data path: {self.test_data_path}")

            # Step 2: Data Transformation
            logging.info("Starting Data Transformation")
            data_transformation = DataTransformation()
            self.train_arr, self.test_arr = data_transformation.initiate_data_transformation(self.train_data_path, self.test_data_path)
            logging.info("Data Transformation completed")

            # Step 3: Model Training
            logging.info("Starting Model Training")
            model_trainer = ModelTrainer()
            self.r2_score = model_trainer.initiate_model_trainer(self.train_arr, self.test_arr)
            logging.info(f"Model Training completed. R2 Score: {self.r2_score}")

        except Exception as e:
            logging.error(f"Error in train pipeline: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()
