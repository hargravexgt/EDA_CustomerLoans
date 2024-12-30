import yaml

def credentials_as_dict():
    with open('file.yaml', 'r') as file:
        data = yaml.safe_load(file) 
    return data



class RDSDatabaseConnector:
    def __init__(self, credentials_func):
        self.credentials_dict = credentials_func()

    def init_engine(self):
        