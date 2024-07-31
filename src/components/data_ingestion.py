import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

sys.path.insert(0, '/Volumes/gyaniData/projects/web-applications/python/mlp1/src')
from exception import CustomException
from logger import logging
from components.data_transformation import DataTransformation
from components.data_transformation import DataTransformationConfirg
from components.model_trainer import modelTrainerConfig
from components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfirg:
    train_data_path : str =os.path.join('artifacts','train.csv')
    test_data_path : str =os.path.join('artifacts','test.csv')
    raw_data_path : str =os.path.join('artifacts','raw.csv')

class DataIngestion :
    def __init__(self) :
        self.ingestion_config=DataIngestionConfirg()

    def initiate_data_ingestion(self) :
        logging.info("Entered the data ingestion method ")
        try :
            #Reading Data from CSV, Can be done from ANY DB
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the data from CSV to dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info('Train Test Split Initated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data= obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)


    modelTrainer= ModelTrainer()
    predictedScore=modelTrainer.inititate_model_trainer(train_arr,test_arr)
    print(predictedScore)
        