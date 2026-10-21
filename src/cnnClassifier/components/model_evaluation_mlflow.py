import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
import os
from cnnClassifier import logger
import requests
import json




class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size =self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self._valid_generator = valid_datagenerator.flow_from_directory(
            directory = self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self._valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        import mlflow
        import os
        from cnnClassifier import logger
        
        logger.info("Savunma sanayisi modunda yerel izole kayit baslatiliyor...")
        
        # 1. Adım: Eski bulut/DagsHub değişkenlerini bellekten kazıyoruz
        for key in list(os.environ.keys()):
            if "MLFLOW" in key or "DAGSHUB" in key:
                del os.environ[key]
                
        # 2. Adım: Hata çıkaran gizli sistem etiketlerini tamamen kapatıyoruz
        os.environ["MLFLOW_TRACKING_DISABLE_SYSTEM_TAGS"] = "true"
        os.environ["MLFLOW_DISABLE_SYSTEM_METRICS_LOGGING"] = "true"
        
        # 3. Adım: Takip veritabanını içine kilitliyoruz
        mlflow.set_tracking_uri("sqlite:///guvenli_takip.db")
        # 4. Adım: Temiz bir run başlatıp listeyi indekslerine göre güvenlice çözüyoruz
        with mlflow.start_run(run_name="kidney_disease_secure_run", nested=False, tags={}):
            mlflow.log_params(self.config.all_params)
            
            # self.score listesinden loss ve accuracy değerlerini ayıklıyoruz
            mlflow.log_metric("eval_loss", float(self.score[0]))
            mlflow.log_metric("eval_accuracy", float(self.score[1]))
            
            mlflow.keras.log_model(self.model, "model")
            logger.info("Tum metrikler yerel izele veritabanina basariyla islendi!")



            
            
    