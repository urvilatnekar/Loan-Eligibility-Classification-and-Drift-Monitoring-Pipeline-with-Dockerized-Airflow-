import datetime
import os
import pandas as pd
from sqlalchemy.sql import text # Imports the text function from the SQLAlchemy library, which is often used for executing SQL queries.

try:
    from src import config
    from src import helpers
    from src import queries
except ImportError:
    import config
    import helpers
    import queries

# helpers.create_table_ml_job()

#Defines a function named extract_data that takes two date parameters, start_date and end_date, with a default value of today's date.
#  It is expected to return a Pandas DataFrame. This function is responsible for extracting data from a database.
def extract_data(start_date:datetime.date, end_date:datetime.date=datetime.date.today()) -> pd.DataFrame:
    
    assert start_date <= end_date, "start_date must be less than end_date" #assert statement checks if start_date is less than or equal to end_date.
    print("[INFO] Extracting data from the database since {0} to {1} ...".format(start_date, end_date))
    helpers.engine.execute(text("""drop table if exists customer;""").execution_options(autocommit=True))
    #Data is retrieved from the database using SQL queries defined in an external module named queries.
    helpers.engine.execute(text(queries.CREATE_TEMP_TABLE_CUSTOMER).execution_options(autocommit=True))
    helpers.engine.execute(text("""drop table if exists loan;""").execution_options(autocommit=True))
    helpers.engine.execute(text(queries.CREATE_TEMP_TABLE_LOAN.format(start_date=start_date, end_date=end_date)).execution_options(autocommit=True))
    helpers.engine.execute(text("""drop table if exists credit;""").execution_options(autocommit=True))
    helpers.engine.execute(text(queries.CREATE_TEMP_TABLE_CREDIT).execution_options(autocommit=True))
    df = pd.read_sql(text(queries.GET_DATA), helpers.engine)
    return df

def collect_data(start_date:datetime.date, end_date:datetime.date=datetime.date.today(), job_id:str=None):
    """
    Defines a function named collect_data that collects data from the database and saves it to a CSV file.

The function takes three parameters: start_date, end_date, and job_id.
assert statements validate the parameter types and relationships.
The function calls extract_data to retrieve data as a DataFrame.
The size of the DataFrame is calculated.
A filename is generated based on job_id, start_date, and end_date.
The save_dataset function from the helpers module is called to save the DataFrame as a CSV file in a specified directory.
The filename is returned.
    """
    assert isinstance(start_date, datetime.date)
    assert isinstance(end_date, datetime.date)
    assert isinstance(job_id, str)
    assert start_date <= end_date
    size = 0

    df = extract_data(start_date, end_date)
    size = df.shape[0]
    filename = os.path.join(config.PATH_DIR_DATA, "raw", f"{job_id}_"+start_date.strftime("%Y-%m-%d")+"_"+end_date.strftime("%Y-%m-%d")+".csv")
    helpers.save_dataset(df, filename)
    return filename

if __name__=="__main__":
    job_id = helpers.generate_uuid()
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date.today()
    collect_data(start_date, end_date, job_id)
