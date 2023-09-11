import traceback
import pandas as pd
import numpy as np
import re
import os
import json
import pickle
import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler


from functools import reduce
try:
    import config
    import helpers
except ImportError:
    import config
    import helpers

"""
Improvements:
    1. Use LabelEncoder to encode categorical variables.
    2. Use LabelBinarizer to encode categorical variables.
    3. Use OrdinalEncoder to encode categorical variables.


"""
cat_vars = list(map(str.lower, config.CAT_VARS)) #list contains the names of categorical variables in lowercase.
num_vars = list(map(str.lower, config.NUM_VARS)) #list contains the names of numerical variables in lowercase.
date_vars = list(map(str.lower, config.DATETIME_VARS)) # list contains the names of date-related variables in lowercase.
exc_vars = list(map(str.lower, config.EXC_VARIABLES)) #list contains the names of excluded variables in lowercase.
engineered_vars = {
    "categorical": ["application_year", "application_month", "application_week", "application_day", "application_season"],
    "numerical": ["current_credit_balance_ratio"],
    "date": ["application_date"]
}

######  missing values ######
def get_variables_with_missing_values(df:pd.DataFrame) -> pd.DataFrame:
    #function is designed to find and return a list of variables (columns) in a pandas DataFrame df that have missing (null) values.
    """
    Get variables with missing values.
    :param df: DataFrame
    :return: DataFrame
    """
    missing_counts = df.isnull().sum()
    return missing_counts[missing_counts>0].index.tolist()

def impute_missing_values(df:pd.DataFrame, method:str="basic", mode:str=None, cat_vars:list=config.CAT_VARS, num_vars:list=config.NUM_VARS, job_id:str="") -> pd.DataFrame:
    """
    df: The DataFrame with missing values that you want to impute.
    method: Specifies the imputation method, which can be either "basic" or "advanced." The default is "basic."
    mode: Specifies the mode in which the function operates, which can be "training" or "inference." The default is None.
    cat_vars: A list of categorical variable names. It is set to the default categorical variables from the config module.
    num_vars: A list of numerical variable names. It is set to the default numerical variables from the config module.
    job_id: A unique identifier for the job. This is used when saving and loading imputation models.
    """
    assert mode in ("training", "inference"), f"mode must be either 'training' or 'inference', but got {mode}"
    assert method in ["basic", "advanced"], f"{method} is not a valid methods (basic, advanced)"
    """
    If mode is "training," the function calculates basic imputation values for missing data based on the specified method.
    It creates an imputation model and saves it as a pickle file.
    For categorical variables, it imputes missing values with the mode (most frequent value) of the respective column.
    For numerical variables, it imputes missing values with the mean of the respective column.
    For datetime variables, it also imputes missing values with the mode.
    It ignores certain variables like "loan_id," "customer_id," "loan_status," and those specified in exc_vars
    If mode is "inference," the function loads a pre-trained imputation model using the job_id parameter and imputes missing values in the DataFrame based on the previously calculated imputation values.
    It imputes missing values for the same set of variables as in the training mode.
"""
    if mode=="training":
        model = {
            "method": method,
            "imputes": dict()
        }
        for col in df.columns:
            print("[INFO] Treating missing values in column:", col)
            model["imputes"][col] = dict()
            if method=="basic":
                if col in set(cat_vars + engineered_vars["categorical"]):
                    model["imputes"][col]['mode'] = df[df[col].notnull()][col].mode()[0]
                elif col in set(num_vars + engineered_vars["numerical"]):
                    model["imputes"][col]['mean'] = df[df[col].notnull()][col].mean()
                elif col in set(config.DATETIME_VARS + engineered_vars["date"]):
                    model["imputes"][col]['mode'] = df[df[col].notnull()][col].mode()[0]
                elif col in ["loan_id", "customer_id", "loan_status"] + exc_vars:
                    pass
                else:
                    raise ValueError(f"[ERROR]{col} is not a valid variable")
            if method=="advanced":
                raise(NotImplementedError)
        helpers.save_model_as_pickle(model, f"{job_id}_missing_values_model")
        return impute_missing_values(df, method=method, mode="inference", cat_vars=cat_vars, num_vars=num_vars, job_id=job_id)
    else:
        model = helpers.load_model_from_pickle(model_name=f"{job_id}_missing_values_model")
        cols = get_variables_with_missing_values(df)
        method = model["method"]
        if method=="basic":
            for col in cols:
                if col in set(cat_vars + engineered_vars["categorical"]):
                    df[col].fillna(model["imputes"][col]['mode'], inplace=True)
                elif col in set(num_vars + engineered_vars["numerical"]):
                    df[col].fillna(model["imputes"][col]['mean'], inplace=True)
                elif col in set(config.DATETIME_VARS + engineered_vars["date"]):
                    df[col].fillna(model["imputes"][col]['mode'], inplace=True)
                elif col in ["loan_id", "customer_id", "loan_status"] + exc_vars:
                    pass
                else:
                    raise ValueError(f"[ERROR]{col} is not a valid variable. Pre-trained vairables: {list(model['imputes'].keys())}")
        if method=="advanced":
            raise(NotImplementedError)
    return df

###### enforcing datatypes ######
def enforce_numeric_to_float(x: str) -> float:
    """
    Convert numeric to float. To ensure that all stringified numbers are converted to float.
    :param x: str
    :return: float
    """
    #The function attempts to convert the input x to a float using float().
    #If the conversion is successful, it returns the float representation.
    #If the conversion fails (e.g., if x contains non-numeric characters), it returns np.nan.
    try:
        return float(re.sub("[^0-9.]","", str(x)))
    except ValueError:
        return np.nan

def enforce_datatypes_on_variables(df:pd.DataFrame, cat_vars:list=[], num_vars:list=[]) -> pd.DataFrame:
    """
    df: The DataFrame whose variable data types you want to enforce.
    cat_vars: A list of categorical variable names that should be transformed.
    num_vars: A list of numerical variable names that should be transformed.
    """
    #It converts the "application_time" column to a datetime data type using pd.to_datetime.
    #For each variable in num_vars, it applies the enforce_numeric_to_float function to convert the variable's values to floats. This ensures that all values are numeric or np.nan if they cannot be converted.
    #For each variable in cat_vars, it converts the variable to a string data type using astype(str).
    df["application_time"] = pd.to_datetime(df["application_time"])
    for var in num_vars:
        df[var] = df[var].apply(lambda x: enforce_numeric_to_float(x))
    for var in cat_vars:
        df[var] = df[var].astype(str)
    return df


###### encoding categorical variables ######
def categorize_years_in_current_job(x: str) -> int:
    """
    The function first strips any leading or trailing whitespace from the input string x using str.strip().
It categorizes the years in the current job based on the following rules:
If x is '< 1 year', it returns 0.
If x is any of '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', or '10 years', it extracts the numeric part using re.sub("[^0-9]", "", x) and converts it to an integer.
If x is '10+ years', it returns 11 to represent ten or more years.
If none of the above conditions are met, it returns -1 to indicate an invalid or unknown value.
    """
    x = str(x).strip()
    if x=='< 1 year':
        return 0
    if x in ('1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10 years'):
        return int(re.sub("[^0-9]", "", x))
    if x=='10+ years':
        return 11
    else:
        return -1

def term_to_int(x:str) -> int:
    """
    The function converts the input x to an integer based on the following mapping:
If x is "short term", it returns 0.
If x is "long term", it returns 1.
If x does not match any of the above conditions, it returns np.nan (a NumPy representation of "Not a Number") to indicate an invalid or unknown term type.
    """
    if x=="short term":
        return 0
    elif x=="long term":
        return 1
    else:
        return np.nan

def home_ownership_to_int(x: str) -> int:
    """
    The function performs the following mappings:
If x is "rent", it returns 0 to represent renting.
If the string "mortgage" is found anywhere within x, it returns 1 to represent owning with a mortgage.
If the string "own" is found anywhere within x, it returns 2 to represent outright ownership.
If none of the above conditions are met, it returns np.nan (a NumPy representation of "Not a Number") to indicate an invalid or unknown home ownership category.
    """
    if x=="rent":
        return 0
    elif "mortgage" in x:
        return 1
    elif "own" in x:
        return 2
    else:
        return np.nan

def train_purpose_to_int_model(x:pd.Series, method:str, job_id:str="") -> dict:
    #function allows you to preprocess and encode the purpose variable into a numerical format using different encoding methods, making it suitable for use in machine learning models or analysis.
    """
    build a model file to be used to convert string variable `purpose` into integer datatype
    :param x:pd.Series
    :param method: str, "ranking", "one-hot", "weighted ranking", "relative ranking"
        For ranking
            rank values by their frequency and assign a rank to each value. The most frequent value will have the highest rank
        For relative ranking 
            replace each value by the ratio of its frequency to the highest frequency
        For weighted ranking
            replace each value by the ratio of its frequency to the total number of values
        For one-hot method
            convert values to one-hot encoded vectors
    :param job_id: str, job id
    :return: dict
    """

    """
    The function starts by calculating the frequency (count) of each unique value in the purpose variable using value_counts().
Depending on the specified encoding method, it performs one of the following operations:
For "ranking" method:
It ranks the values based on their frequency, where the most frequent value is assigned the highest rank, the second most frequent value is assigned the second-highest rank, and so on.
It then creates a dictionary that maps each unique value to its rank.
For "relative ranking" method:
It calculates the ratio of the frequency of each value to the highest frequency. This represents the relative ranking of each value.
It creates a dictionary mapping each unique value to its relative ranking.
For "weighted ranking" method:
It calculates the ratio of the frequency of each value to the total number of values. This represents the weighted ranking of each value.
It creates a dictionary mapping each unique value to its weighted ranking.
The resulting mapping is then saved as a JSON model file with a name that includes the job_id.
    """
    assert method in ["ranking", "weighted ranking", "relative ranking"], f"{method} is not a valid methods (ranking, one-hot, weighted ranking, relative ranking)"
    val_counts = x.value_counts()
    if method=="ranking":
        uniq_vals = sorted(val_counts.unique(), reverse=False)
        val_to_int = dict(zip(uniq_vals, range(1, len(uniq_vals)+1)))
        model = val_counts.apply(lambda x: val_to_int[x]).to_dict()
        helpers.save_model_as_json(model, f"{job_id}_purpose_to_int_model")        
        return model
    if method=="relative ranking":
        model = (val_counts/val_counts.max()).to_dict()
        helpers.save_model_as_json(model, f"{job_id}_purpose_to_int_model")        
        return model
    if method=="weighted ranking":
        model = (val_counts/val_counts.sum()).to_dict()
        helpers.save_model_as_json(model, f"{job_id}_purpose_to_int_model")        
        return model

def purpose_to_int(x:pd.Series, mode:str, method:str=None, model:str=None, job_id:str="") -> pd.Series:
    """
    Convert purpose to int.
    :param x:pd.Series
    :param mode: str, choose from "training", "inference"
    :param method: str, "ranking",  "weighted ranking", "relative ranking"
        For ranking method
            rank values by their frequency and assign a rank to each value. The most frequent value will have the highest rank
        For relative ranking
            replace each value by the ratio of its frequency to the highest frequency
        For weighted ranking
            replace each value by the ratio of its frequency to the total number of values
        For one-hot method
            convert values to one-hot encoded vectors
        when method is None and model is not None, any new value (not present in the model) will be encoded as 0
    :param model: method, model to predict the purpose. If None, a new model will be trained and saved to the default directory of models as defined in the config file
    :param save_report: bool, whether to save the report of missed/new values. Not implemented for nor
    :param job_id: str, job id
    :return:pd.Series
    """

    """
The function first checks the mode parameter to determine whether it's in training or inference mode.
In training mode (mode=="training"):
If no model is provided (model==None), it trains a new model using the train_purpose_to_int_model function, which is explained in a previous response.
It then applies the encoding method specified by method to each value in the input Series x using the trained model.
In inference mode (mode=="inference"):
It loads a pre-trained model (specified by model) for encoding purposes. The pre-trained model should have been saved during training.
It applies the encoding method specified by method to each value in the input Series x using the pre-trained model. If a value is not found in the model, it is encoded as 0.
    """
    print("[INFO] Converting purpose to int using method:", method)
    if model==None:
        print("[INFO] No model for purpose-to-int conversion provided. Training a new model first...")
        mode = "training"
    if mode=="training":
        model = train_purpose_to_int_model(x, method, job_id=job_id)
        # return purpose_to_int(x, method=method, model=model, job_id=job_id)
        return x.apply(lambda x: model.get(x, 0))
    else:
        model = helpers.load_model_from_json(model_name=f"{job_id}_purpose_to_int_model")
        return x.apply(lambda x: model.get(x, 0))

def loan_status_to_int(x: str) -> int: #function is used to convert loan status values (e.g., "loan given" or "loan refused") into integer representations. 
    """
    Convert loan status to int.
    :param x: str, lower cased loan status
    :return: int
    """
    #The function checks the input x to determine the loan status.
    #If x is "loan refused" (case-insensitive), it returns 0 to represent a loan refusal.
    #If x is "loan given" (case-insensitive), it returns 1 to represent a loan approval.
    #If x is not one of these values or an integer, it raises an assertion error.

    assert x in ("loan given", "loan refused") or isinstance(x, int), f"{x} is not a valid loan status and is not an integer"
    if x.strip()=="loan refused":
        return 0
    if x.strip()=="loan given":
        return 1
    return x

def encode_categorical_variables(df:pd.DataFrame, mode="training", purpose_encode_method="ranking", job_id:str="") -> pd.DataFrame:
    """
    Encode categorical variables.
    :param df: DataFrame
    :param mode: str, "training" or "inference"
    :param purpose_encode_method: str, choose from "ranking", "weighted ranking", "relative ranking"
    :param job_id: str, job id
    :return: DataFrame
    """
#The function first checks the mode parameter to determine whether it's in "training" or "inference" mode.
#For categorical variables listed in config.CAT_VARS, it converts the values to lowercase to ensure consistent encoding.
#The "term" column is converted to integer values using the term_to_int function.
#The "home_ownership" column is converted to integer values using the home_ownership_to_int function.
#The "years_in_current_job" column is categorized into integer values using the categorize_years_in_current_job function.
#If the target variable (specified in config.TARGET) is present in the DataFrame, it is encoded as an integer using the loan_status_to_int function.
#The "purpose" column is encoded using the purpose_to_int function based on the specified encoding method and mode.

    assert mode in ("training", "inference"), f"{mode} is not a valid mode (training , inference)"
    assert isinstance(job_id, str)
    for col in config.CAT_VARS:
        assert col in df.columns, f"{col} not in {df.columns}"
        df[col] = df[col].str.lower()

    df["term"] = df["term"].apply(lambda x: term_to_int(x))
    df["home_ownership"] = df["home_ownership"].apply(lambda x: home_ownership_to_int(x))  
    df["years_in_current_job"] = df["years_in_current_job"].apply(lambda x: categorize_years_in_current_job(x))
    if config.TARGET.lower() in df.columns:
        df[config.TARGET.lower()] = df[config.TARGET.lower()].apply(lambda x: loan_status_to_int(x))
    df["purpose"] = purpose_to_int(df["purpose"], mode=mode, method=purpose_encode_method, job_id=job_id)
    return df

###### engineer new variables ######
def month_to_season(month:int) -> int:
    """
    Convert date to season.
    :param m: int, month between 1 and 12
    :return: int
    """
#The function takes the input month and assigns it to one of the four seasons based on common season-month associations:
#Months 1, 2, and 3 are assigned to season 1 (winter).
#Months 4, 5, and 6 are assigned to season 2 (spring).
#Months 7, 8, and 9 are assigned to season 3 (summer).
#Months 10, 11, and 12 are assigned to season 4 (autumn or fall).
#If the input month is not within the range of 1 to 12, the function returns np.nan (NumPy's representation of a missing or undefined value).
    if month in [1, 2, 3]:
        return 1
    elif month in [4, 5, 6]:
        return 2
    elif month in [7, 8, 9]:
        return 3
    elif month in [10, 11, 12]:
        return 4
    else:
        return np.nan

def engineer_variables(df:pd.DataFrame) -> pd.DataFrame:
    """
    Engineer variables.
    :param df: DataFrame
    :return: DataFrame
    """
#The function performs several transformations on the input DataFrame to create new features:
#application_date: It extracts the date component from the application_time column to create a new application_date column containing only the date.
#application_year: It extracts the year component from the application_time column.
#application_month: It extracts the month component from the application_time column.
#application_week: It extracts the week of the year from the application_time column.
#application_day: It extracts the day of the month from the application_time column.
#application_season: It uses the month_to_season function to map the application_month to a season (1 for winter, 2 for spring, 3 for summer, 4 for autumn).
#current_credit_balance_ratio: It calculates the ratio of current_credit_balance to current_loan_amount and fills missing values with 0.0.
#The engineered variables are added to the DataFrame, and the modified DataFrame is returned.

#This function is useful for creating additional features from existing data, which can improve the performance of machine learning models or provide valuable insights during data analysis.

    for col in ["application_time"]:
        assert col in df.columns, f"{col} not in {df.columns}"

    df["application_date"] = df["application_time"].dt.date
    df["application_year"] = df["application_time"].dt.year
    df["application_month"] = df["application_time"].dt.month
    df["application_week"] = df["application_time"].dt.week
    df["application_day"] = df["application_time"].dt.day
    df["application_season"] = df["application_month"].apply(lambda x: month_to_season(x))
    df["current_credit_balance_ratio"] = (df["current_credit_balance"]/df["current_loan_amount"]).fillna(0.0)
    return df

def split_train_test(df:pd.DataFrame, test_size:float, method:str='time based'):
    """
    Split data into train and test.
    :param df: DataFrame
    :param test_size: float, between 0 and 1
    :param method: str, 'time based' or 'random'
    :return: (DataFrame, DataFrame)
    """
#If the method is 'random':
#The function shuffles the rows of the input DataFrame using a random seed defined in config.RANDOM_STATE.
#It then splits the shuffled DataFrame into two parts based on the specified test_size. The first part is the testing set, and the second part is the training set.
#The function returns the training and testing DataFrames.
#If the method is 'time based':

#The function assumes that the DataFrame contains a column named "application_date," which represents the date of each record.
#It sorts the unique dates in ascending order.
#It splits the data such that the training set includes data up to a certain date (determined by test_size), and the testing set includes data after that date.
#The function returns the training and testing DataFrames.
#Error Handling:

#If an invalid method is provided, the function raises a ValueError.
    if method=='random':
        return df.sample(frac=1, random_state=config.RANDOM_STATE).iloc[:int(len(df)*test_size)], df.sample(frac=1, random_state=config.RANDOM_STATE).iloc[int(len(df)*test_size):]
    if method=='time based':
        unique_dates = sorted(df["application_date"].unique())
        
        train_dates = unique_dates[:int(len(unique_dates)*(1-test_size))]
        test_dates = unique_dates[unique_dates.index(train_dates[-1])+1:]
        train_df = df[df["application_date"].isin(train_dates)]
        test_df = df[df["application_date"].isin(test_dates)]

        return train_df, test_df
    raise(ValueError(f"{method} is not a valid method (time based, random)"))

def rescale_data(df:pd.DataFrame, method:str='standardize', mode:str='training', columns:list=[], job_id:str="") -> pd.DataFrame:
    """
    Rescale data.
    :param df: DataFrame
    :param method: str, 'standardize' or 'minmax'
    :param mode: str, 'training' or 'inference'
    :return: DataFrame
    """
#If the mode is 'training':

#The function fits a scaler to the specified columns based on the chosen rescaling method ('standardize' or 'minmax'). The scaler is either a StandardScaler or a MinMaxScaler.
#It saves the scaler and other relevant information as a model using helpers.save_model_as_pickle. The model includes the scaler and the rescaling method used.
#It then applies the scaler to the specified columns and adds the rescaled columns to the DataFrame with names prefixed by the rescaling method.
#If the mode is 'inference':

#The function loads the scaler and rescaling method from the previously saved model using helpers.load_model_from_pickle.
#It checks if the specified columns can be converted to float; if not, it prints a debug message for skipped columns.
#It applies the loaded scaler to the specified columns and adds the rescaled columns to the DataFrame with names prefixed by the rescaling method.
#Error Handling:

#The function performs checks to ensure that the provided method and mode are valid and that the specified columns exist in the DataFrame.
#If any of these checks fail, the function raises AssertionError.
    assert method in ('standardize', 'minmax'), f"{method} is not a valid method (standardize, minmax)"
    assert mode in ('training', 'inference'), f"{mode} is not a valid mode (training, inference)"
    for col in columns:
        assert col in df.columns

    if mode=='training':
        if method=='standardize':
            scaler = StandardScaler()
            scaler.fit(df[columns])
        if method=='minmax':
            scaler = MinMaxScaler()
            scaler.fit(df[columns])
        model = {
            'scaler': scaler,
            'method': method,
        }

        helpers.save_model_as_pickle(model, f"{config.PATH_DIR_MODELS}/{job_id}_numerical_scaler.pkl")
        df[list(map(lambda x: f"{method}_{x}", columns))] = scaler.transform(df[columns])
        return df
    if mode=='inference':
        model = helpers.load_model_from_pickle(model_name=f"{job_id}_numerical_scaler.pkl")
        scaler = model['scaler']
        method = model['method']
        for col in columns:
            try:
                df[col].astype(float)
            except:
                print("[DEBUG] Column skipped:", col)
        df[list(map(lambda x: f"{method}_{x}", columns))] = scaler.transform(df[columns])
        return df

def preprocess_data(df:pd.DataFrame, mode:str, job_id:str=None, rescale=False, ref_job_id:str=None) -> pd.DataFrame:
    """
    Pre-process data and save preprocessed datasets for later use.
    :param df: DataFrame
    :param mode: str, 'training' or 'inference'
    :param job_id: str, job_id for the preprocessed dataset
    :param rescale: bool, whether to rescale data.
    :param ref_job_id: str, job_id of the last deployed model. Usefull when doing inference.
    :return: DataFrame
    """
#The function starts by performing basic data cleaning by dropping rows with null values in specific columns (customer_id, loan_id, loan_status).
#It ensures that column names are in lowercase for consistency.
#If the mode is 'training':
#It checks that the target variable (config.TARGET) exists in the DataFrame.
#It splits the data into a training set (train_df) and a test set (test_df) based on the specified split method (config.SPLIT_METHOD) and the test split size (config.TEST_SPLIT_SIZE).
#It encodes categorical variables and imputes missing values for the training set (train_df).
#If rescale is True, it rescales the numerical data columns using the specified rescaling method (config.RESCALE_METHOD) for the training set (train_df).
#It saves the preprocessed training dataset as a CSV file.
#It then recursively calls preprocess_data with mode="inference" on the test set (test_df) to preprocess it.
#If the mode is 'inference':
#It encodes categorical variables and imputes missing values for the entire DataFrame.
#If rescale is True, it rescales the numerical data columns using the specified rescaling method (config.RESCALE_METHOD) for the entire DataFrame.
#It saves the preprocessed inference dataset as a CSV file.
#It returns the preprocessed DataFrame.
#Error Handling:

#The function checks that the provided mode is valid ('training' or 'inference').
#In 'training' mode, it ensures that the target variable exists in the DataFrame.
#The function provides optional checks for rescaling, which can be enabled or disabled as needed.
    assert mode in ('training', 'inference')
    
    if mode=='training':
        assert config.TARGET in df.columns, f"{config.TARGET} not in {df.columns}"

    df.columns = list(map(str.lower, df.columns))
    initial_size = df.shape[0]
    df = df[df["customer_id"].notnull() & df["loan_id"].notnull() & df["loan_status"].notnull()]
    if mode=='training':
        df["loan_status"] = df["loan_status"].str.lower()
    if df.shape[0] != initial_size:
        print(f"[WARNING] Dropped {initial_size - df.shape[0]} rows with null values in (customer_id, loan_id, loan_status)")
    df = enforce_datatypes_on_variables(df, cat_vars=config.CAT_VARS, num_vars=config.NUM_VARS)
    df = engineer_variables(df)
    if mode=='training':
        # split train and test data before encoding categorical variables and imputing missing values
        train_df, test_df = split_train_test(df, config.TEST_SPLIT_SIZE, method=config.SPLIT_METHOD)
        train_df = encode_categorical_variables(train_df, mode="training", purpose_encode_method=config.PURPOSE_ENCODING_METHOD, job_id=job_id)
        train_df = impute_missing_values(train_df, method="basic", mode="training", job_id=job_id)
        if rescale:
            train_df = rescale_data(train_df, method=config.RESCALE_METHOD, mode="training", columns=num_vars + engineered_vars["numerical"])
        helpers.save_dataset(train_df, os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_training.csv"))
        preprocess_data(test_df, mode="inference", job_id=job_id, ref_job_id=job_id)
    else:
        # if mode is infer, no need to split train and test data
        test_df = encode_categorical_variables(df, mode="inference", purpose_encode_method=config.PURPOSE_ENCODING_METHOD, job_id=ref_job_id)
        test_df = impute_missing_values(test_df, method="basic", mode="inference", job_id=ref_job_id)
        if rescale:
            test_df = rescale_data(test_df, method=config.RESCALE_METHOD, mode="inference", columns=num_vars + engineered_vars["numerical"])
        helpers.save_dataset(test_df, os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_inference.csv"))
    return test_df


if __name__=='__main__':
    preprocess_data(df=helpers.load_dataset(os.path.join(config.PATH_DIR_DATA, "raw", "loan eligibility data", "LoansTraining.csv")), mode="training")
