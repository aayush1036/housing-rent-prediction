import mysql.connector
import joblib 
import os 
import json
import numpy as np 
# Constants 
CITIES = ['AHEMDABAD','BANGALORE','CHENNAI','DELHI','HYDERABAD','KOLKATA','MUMBAI','PUNE']
MODEL_PATH = os.path.join('Objects','Models')
ORDINAL_PATH = os.path.join('Objects','Encoders','OrdinalEncoder')
NUM_COLS = ['bedroom', 'area', 'bathroom']
ALL_COLS = ['seller_type','bedroom','layout_type','property_type','locality','area','furnish_type','bathroom']

paths = {
    'locality':os.path.join('Objects','Encoders','LabelEncoder'),
    'furnish_type':os.path.join(ORDINAL_PATH, 'furniture_encoders'),
    'layout_type':os.path.join(ORDINAL_PATH, 'layout_type'),
    'property_type': os.path.join(ORDINAL_PATH, 'property_type'),
    'seller_type':os.path.join(ORDINAL_PATH, 'seller_type')
}
CONFIG_PATH = 'config.json'

def loadModels()->dict:
    """Loads models for all cities

    Returns:
        dict: Dictionary containing models for all cities
    """    
    model_dict = {city:joblib.load(os.path.join(MODEL_PATH, f'{city}_model.pkl')) for city in CITIES}
    return model_dict

def loadEncoders()->dict:
    """Loads the encoders for every column of every city

    Returns:
        dict: The nested dictionary containing encoders for each column for each city
    """    
    encoder_dict = {city: {col:joblib.load(os.path.join(paths[col], f'{city}_{col}_encoder.pkl')) for col in paths.keys()} for city in CITIES}
    return encoder_dict

def getDataFromForm(request, contribute=False)->dict:
    """Gets data from a form and loads it

    Args:
        request (flask form request): The request from the flask form from where data is to be fetched
        contribute (bool, optional): Whether the data will be used to contribute to the database or not. Defaults to False.

    Returns:
        dict: The dictionary containing column names and data 
    """
    city = request.form['city'].strip()
    values = []
    for col in ALL_COLS:
        if col in NUM_COLS:
            values.append(float(request.form[col].strip()))
        else:
            values.append(request.form[col].strip())
    if contribute:
        ALL_COLS.insert(5, 'price')
        price = float(request.form['price'].strip())
        values.insert(5,price)
    VALUES_DICT = dict(zip(ALL_COLS, values))
    if ALL_COLS[5] == 'price':
        ALL_COLS.pop(5)
    VALUES_DICT['locality'] = VALUES_DICT['locality'].upper()
    return city,VALUES_DICT

def createConnection(config_path=CONFIG_PATH)->mysql.connector.connection:
    """Creates a connection to the database

    Args:
        config_path (str, optional): The path containing the config file. Defaults to CONFIG_PATH.

    Returns:
        mysql.connector.connection: The connection to the database
    """
    with open(config_path, 'r') as f:
        config = json.load(f)    
    
    conn = mysql.connector.connect(
        host=config.get('host'),
        port=config.get('port'),
        user=config.get('user'),
        password=config.get('password'),
        auth_plugin=config.get('auth_plugin'),
        database=config.get('database')
    )
    return conn

def transformData(city:str, data:dict,encoders:dict)->np.array:
    """Transforms the data to make it suitable for prediction

    Args:
        city (str): The city for which we are transforming the data
        data (dict): The dictionary containing column names and data for that particular city
        encoders (dict): The dictionary containing the encoders for all the columns for that city

    Returns:
        np.array: An array containing the required data which can be passed to the model
    """
    transformed_data = []
    for name, value in data.items():
        if name not in NUM_COLS:
            if name == 'locality':
                transformed_data.append(encoders[city][name].transform([value]))
            else:
                transformed_data.append(encoders[city][name].transform([[value]]))
        else:
            transformed_data.append(value)
    return np.array(transformed_data, dtype='object').reshape(-1,1).T