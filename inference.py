import pandas as pd
import numpy as np
import datetime
import os
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaseEnsemble
try:
    from src import config
    from src import helpers
    from src import preprocess
    from src import etl
except:
    import config
    import helpers
    import preprocess
    import etl

def load_model():
    #function loads a machine learning model for making predictions.
    #It checks if the "deploy_report.json" file exists in the model directory.
    #If not, it creates a default JSON with a "prediction_model" key set to None.
    #It then reads the "prediction_model" filename from the JSON and loads the corresponding model from the model directory.
    if not os.path.isfile(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json")):
        json.dump({"prediction_model": None}, open(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json"), "w"))
    filename = json.load(open(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json"), "r"))["prediction_model"]
    with open(os.path.join(config.PATH_DIR_MODELS, filename), "rb") as f:
        model = pickle.load(f)
    return model #Returns the loaded machine learning model.
model = load_model() #load_model function to load the machine learning model and assigns it to the model variable.

def batch_inference(job_id:str, ref_job_id:str, start_date:datetime.date, end_date:datetime.date=datetime.date.today(), predictors=[]) -> dict:
    #function performs batch inference on a dataset using a trained machine learning model.
    
    """
    job_id (str): The unique identifier for the current job.
    ref_job_id (str): The reference job ID that corresponds to the training job used for the model.
    start_date (datetime.date): The start date for data extraction.
    end_date (datetime.date, optional): The end date for data extraction (default is today's date).
    predictors (list): A list of predictor variables used for inference.
    """
    #It retrieves the model type based on the reference job ID.
    #It collects data for inference, preprocesses it, and loads the preprocessed data.
    #It makes predictions using the loaded model and returns a dictionary with loan IDs as keys and prediction values as values.

    model_type = helpers.get_model_type(ref_job_id)
    model = helpers.load_model_from_pickle(f"{ref_job_id}_{model_type}")
    etl.collect_data(start_date=start_date, end_date=end_date, job_id=job_id)
    df = helpers.load_dataset(helpers.locate_raw_data_filename(job_id))
    preprocess.preprocess_data(df, mode="inference", job_id=job_id, ref_job_id=ref_job_id, rescale=False, job_date=start_date, inner_call=False)
    _, test_filename = helpers.locate_preprocessed_filenames(job_id)
    df = helpers.load_dataset(test_filename)
    df['prediction'] = model.predict(df[predictors])
    return dict(df[['loan_id', 'prediction']].values) #Returns a dictionary with loan IDs as keys and model predictions as values.

def make_predictions(df:pd.DataFrame, model:BaseEnsemble=None, predictors=[]) -> pd.DataFrame: #function makes predictions using a machine learning model.
    """
    df (pd.DataFrame): The input DataFrame containing the data to make predictions on.
    model (BaseEnsemble, optional): The machine learning model to use for predictions (default is None, which loads the model using load_model).
    predictors (list): A list of predictor variables used for prediction.
    """
    #It checks if a model is provided; if not, it loads the model.
    #It makes predictions using the model on the input DataFrame and converts the prediction values to human-readable labels ("loan given" or "loan refused").
    if model is None:
        model = load_model()
    if model==None:
        return {"error": "No model deployed"}
    df['prediction'] = model.predict(df[predictors])
    df['prediction'] = df['prediction'].apply(lambda x: "loan given" if x==1 else "loan refused")
    return df[['loan_id', 'prediction']].to_dict(orient='records')#Returns a DataFrame with loan IDs and corresponding prediction labels.

if __name__=='__main__':
    predictors = config.PREDICTORS
    job_date = datetime.date(2016, 11, 1)
    job_id = helpers.generate_uuid()
    ref_job_id = helpers.get_latest_training_job_id(status="pass")
    preds = batch_inference(job_id, ref_job_id, job_date, predictors=predictors)
    print(preds)

# this code demonstrates how to load a machine learning model, perform batch inference on data, and make predictions using the loaded model.
    
    