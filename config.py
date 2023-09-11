RUN_LOCAL = False # is a boolean variable that determines whether the code should run in a local environment 
if RUN_LOCAL:
    PATH_DIR_DATA = "C:/data" 
    PATH_DIR_MODELS = ".C:/models"
    PATH_DIR_RESULTS = "C:/results"
    PATH_TO_CREDENTIALS = "C:/Creds.json"
    PATH_TO_APP_SHELL = "C:/app.sh"
else:
    PATH_DIR_DATA = "C:/data"
    PATH_DIR_MODELS = "C:/models"
    PATH_DIR_RESULTS = "C:/results"
    PATH_TO_CREDENTIALS = "C:/Creds.json"
    PATH_TO_APP_SHELL = "C:/app.sh"

RANDOM_SEED = 42 #This line sets the random seed to 42. A random seed is used to initialize the random number generator, ensuring that the same sequence of random numbers is generated each time the code is run. It's often used to make experiments reproducible.
TEST_SPLIT_SIZE = 0.3
PROB_THRESHOLD = 0.5
SPLIT_METHOD = "time based" #This line specifies the data split method as "time based." It suggests that the data is split into training and testing sets based on a temporal criterion, such as using older data for training and more recent data for testing.

# lowest acceptable difference between the performances of the same model on two different datasets
MODEL_DEGRADATION_THRESHOLD = 0.1 #This line sets the threshold for model degradation to 0.1. It likely represents the lowest acceptable difference in model performance between two different datasets.
ASSOCIATION_DEGRADATION_THRESHOLD = 0.3 #This line sets the threshold for association degradation to 0.3. It may be used to determine whether a change in the dataset significantly impacts model performance.

# lowest acceptable performance of either accuracy, precision, recall, f1 or auc depending on the classification usecase
MODEL_PERFORMANCE_THRESHOLD = 0.7 #This line sets the threshold for model performance to 0.7. It represents the lowest acceptable value for a model performance metric (e.g., accuracy, precision, recall, F1-score, or AUC).
MODEL_PERFORMANCE_METRIC = "auc" #This line specifies the model performance metric as the area under the ROC curve (AUC). 

IDENTIFIERS = ['loan_id', 'customer_id'] #This line defines a list of identifier columns in the dataset, such as 'loan_id' and 'customer_id'. These columns are typically not used as predictors but serve to uniquely identify each data point.
TARGET = 'loan_status' #This line specifies the target variable for the analysis, which is 'loan_status'. This variable is typically what the model aims to predict.
DATETIME_VARS = ['application_time']
EXC_VARIABLES = [
    'application_time'
    ]
#The subsequent lines define lists of categorical (CAT_VARS) and numerical (NUM_VARS) variables, predictor variables (PREDICTORS), stages in the analysis pipeline (STAGES), status categories (STATUS), and job types (JOB_TYPES).
PURPOSE_ENCODING_METHOD = "weighted ranking" # choose from (ranking, one-hot, weighted ranking, relative ranking)
RESCALE_METHOD = "standardize" # choose from (standardize, minmax, None)
CAT_VARS = [
    'term', 
    'home_ownership', 
    'purpose',
    'years_in_current_job', 
    ]
NUM_VARS = [
    'current_loan_amount', 
    'credit_score', 
    'monthly_debt',
    'annual_income',
    'years_of_credit_history', 
    'months_since_last_delinquent', 
    'no_of_open_accounts',
    'current_credit_balance',
    'max_open_credit',
    'bankruptcies',
    'tax_liens', 
    'no_of_properties', 
    'no_of_cars',
    'no_of_children', 
    'no_of_credit_problems', 
    ]

PREDICTORS = [
    "current_loan_amount",
    "term",
    "credit_score",
    "years_in_current_job",
    "home_ownership",
    "annual_income",
    "purpose",
    "monthly_debt",
    "years_of_credit_history",
    "months_since_last_delinquent",
    "no_of_open_accounts",
    "no_of_credit_problems",
    "current_credit_balance",
    "max_open_credit",
    "bankruptcies",
    "tax_liens",
    'no_of_properties', 
    'no_of_cars',
    'no_of_children',
    "application_year",
    "application_month",
    "application_week",
    "application_day",
    "application_season",
    "current_credit_balance_ratio",
]

STAGES = [
    "etl", "preprocess", "training", "testing", "inference", "postprocess", "preprocess-training", "preprocess-inference", "report", "driftcheck",
    "etl_report", "raw_data_drift-report", "deploy"
    ]
STATUS = ["pass", "fail", "skipped", "started"]
JOB_TYPES = ["training", "inference", None]


