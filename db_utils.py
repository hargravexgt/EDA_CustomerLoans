import yaml
from sqlalchemy import create_engine, text
import pandas as pd

def load_credentials(filepath='credentials.yaml'):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file) 
    return data

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


def save_df_as_csv(df, filename):
    df.to_csv(filename)
    print("File saved.")

def read_csv_as_df(filename):
    df = pd.read_csv(filename)
    return df

if __name__ == "__main__":
    credentials = load_credentials()
    connector = RDSDatabaseConnector(credentials)
    connector.init_engine()
    loan_payments = connector.extract_loan_payments_data()
    save_df_as_csv(loan_payments, 'loan_payments.csv')
    loan_payments.head()

