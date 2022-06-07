import mysql.connector
import joblib 
import os 
import json

# Constants 
CITIES = ['AHEMDABAD','BANGALORE','CHENNAI','DELHI','HYDERABAD','KOLKATA','MUMBAI','PUNE']
MODEL_PATH = os.path.join('Objects','Models')
ORDINAL_PATH = os.path.join('Objects','Encoders','OrdinalEncoder')
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

def loadEncoders(col:str)->dict:
    """Loads the encoders for particular columns

    Args:
        col (str): The column for which you want to load the encoders

    Returns:
        dict: The dictionary containing encoders for that column for all cities
    """
    base_path = paths[col]
    encoder_dict = {city:joblib.load(os.path.join(base_path, f'{city}_{col}_encoder.pkl')) for city in CITIES}
    return encoder_dict

def getDataFromForm(request, contribute=False)->tuple:
    """Gets data from a form and loads it

    Args:
        request (flask form request): The request from the flask form from where data is to be fetched
        contribute (bool, optional): Whether the data will be used to contribute to the database or not. Defaults to False.

    Returns:
        tuple: The tuple containing the values
    """
    city = request.form['city'].strip()
    seller_type = request.form['seller_type'].strip()
    bedrooms = int(request.form['bedroom'])
    layout_type = request.form['layout_type'].strip()
    property_type = request.form['property_type'].strip()
    locality = request.form['locality'].strip().upper()
    area = float(request.form['area'])
    furnish_type = request.form['furnish_type'].strip()
    bathroom = int(request.form['bathroom'])
    values = [seller_type, bedrooms, layout_type, property_type, locality, area, furnish_type, bathroom]
    if contribute:
        price = float(request.form['price'].strip())
        values.insert(5,price)
    return city, tuple(values)

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