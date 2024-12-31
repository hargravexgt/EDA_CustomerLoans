import yaml
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

def load_credentials(filepath='credentials.yaml'):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file) 
    return data

def save_df_as_csv(df, filename):
    df.to_csv(filename)
    print("File saved.")

def read_csv_as_df(filename):
    df = pd.read_csv(filename)
    return df

class RDSDatabaseConnector:
    def __init__(self, credentials):
        self.credentials = credentials

    def init_engine(self):
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        USER = self.credentials['RDS_USER']
        PASSWORD = self.credentials['RDS_PASSWORD']
        HOST = self.credentials['RDS_HOST']
        PORT = self.credentials['RDS_PORT']
        DATABASE = self.credentials['RDS_DATABASE']
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
        self.engine = engine

    def extract_loan_payments_data(self):
        with self.engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM loan_payments"))
            return pd.DataFrame(result)

class DataTransform:
    def __init__(self):
        pass
        
    def convert_col_to_months(self, df, cols = []):
        def convert_elements_to_months(value):
            if pd.isna(value):  # Handle missing values
                return 0
            if 'month' in value:
                months = int(value.split()[0])  # Extract the numeric part
                return months
            elif 'year' in value:
                years = int(value.split()[0])  # Extract the numeric part
                months = years * 12  # Convert years to months
                return months
            else:
                raise ValueError(f"Unknown time format: {value}")
            
        return df[cols].map(convert_elements_to_months)
    
    def convert_col_to_datetime(self, df, cols = []):
        new_cols = pd.DataFrame()
        for col in cols:
            new_cols[col] = pd.to_datetime(df[col], format='%b-%Y', errors='coerce').fillna(
                pd.to_datetime(df[col], format='%B-%Y', errors='coerce'))
            new_cols[col] = new_cols[col].dt.to_period('M') 
        return new_cols
    
    def convert_col_to_float(self, df, cols = []):
        new_cols = pd.DataFrame()
        for col in cols:
            new_cols[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
        return new_cols

class DataFrameInfo:

    def col_dtypes(self, df, cols = None):
        if cols == None:
            return df.dtypes
        else:
            return df[cols].dtypes
        
    def col_stats(self, df, cols = None):
        if cols == None:
            return df.agg(['median', 'mean', 'std'])
        else:
            return df[cols].agg(['median', 'mean', 'std'])
        
    def unique_values(self, df, cols = None):
        if cols == None:
            df = df
        else:
            df = df[cols]
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool'])
        unique_counts = {col: df[col].nunique() for col in categorical_cols.columns}
        return unique_counts
    
    def print_shape(self, df, cols=None):
        if cols == None:
            df = df
        else:
            df = df[cols]
        print(df.shape)
        return df.shape
    
    def col_null_percentages(self, df, cols=None):
        if cols == None:
            df = df
        else:
            df = df[cols]
        return df.isnull().sum()/len(df)
    
class DataFrameTransformer:

    def impute_col_with_mean(self, df, col):
        df[col] = df[col].fillna(df[col].mean())
    
    def impute_col_with_median(self, df, col):
        df[col] = df[col].fillna(df[col].median())


if __name__=="__main__":
    credentials = load_credentials()
    connector = RDSDatabaseConnector(credentials)
    connector.init_engine()
    loan_payments = connector.extract_loan_payments_data()
    save_df_as_csv(loan_payments, 'loan_payments.csv')
    loan_payments.head()

