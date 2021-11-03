from threading import local
from flask import Flask, render_template,request
import joblib
import numpy as np
import os 

# CONSTANTS 
CITIES = ['AHEMDABAD','BANGALORE','CHENNAI','DELHI','HYDERABAD','KOLKATA','MUMBAI','PUNE']
MODEL_PATH = 'Objects/Models/'
LOCALITY_ENCODER_PATH = 'Objects/Encoders/LabelEncoder/'
FURNITURE_ENCODER_PATH = 'Objects/Encoders/OrdinalEncoder/furniture_encoders/'
LAYOUT_ENCODER_PATH = 'Objects/Encoders/OrdinalEncoder/layout_type/'
PROPERTY_ENCODER_PATH = 'Objects/Encoders/OrdinalEncoder/property_type/'
SELLER_ENCODER_PATH = 'Objects/Encoders/OrdinalEncoder/seller_type/'

# Loading the encoders 
model_dict = {}
locality_encoder_dict = {}
furniture_encoder_dict = {}
layout_encoder_dict = {}
property_encoder_dict = {}
seller_encoder_dict = {}

for city in CITIES:
    model_dict[city] = joblib.load(os.path.join(MODEL_PATH, f'{city}_model.pkl'))
    locality_encoder_dict[city] = joblib.load(os.path.join(LOCALITY_ENCODER_PATH, f'{city}_locality_encoder.pkl'))
    furniture_encoder_dict[city] = joblib.load(os.path.join(FURNITURE_ENCODER_PATH, f'{city}_furnish_type_encoder.pkl'))
    layout_encoder_dict[city] = joblib.load(os.path.join(LAYOUT_ENCODER_PATH, f'{city}_layout_type_encoder.pkl'))
    property_encoder_dict[city] = joblib.load(os.path.join(PROPERTY_ENCODER_PATH, f'{city}_property_type_encoder.pkl'))
    seller_encoder_dict[city] = joblib.load(os.path.join(SELLER_ENCODER_PATH, f'{city}_seller_type_encoder.pkl'))

# Create a flask app
app = Flask(__name__)

# create the home endpoint
@app.route('/')
def home():
    return render_template('home.html')
#create ahemdabad endpoint
@app.route('/Ahemdabad')
def ahemdabad():
    return render_template('ahemdabad.html')
#create bangalore endpoint
@app.route('/Bangalore')
def bangalore():
    return render_template('bangalore.html')
#create chennai endpoint
@app.route('/Chennai')
def chennai():
    return render_template('chennai.html')
#create delhi endpoint
@app.route('/Delhi')
def delhi():
    return render_template('delhi.html')
# create hyderabad endpoint
@app.route('/Hyderabad')
def hyderabad():
    return render_template('hyderabad.html')
#create kolkata endpoint
@app.route('/Kolkata')
def kolkata():
    return render_template('kolkata.html')
# create mumbai endpoint
@app.route('/Mumbai')
def mumbai():
    return render_template('mumbai.html')
# create pune endpoint 
@app.route('/Pune')
def pune():
    return render_template('pune.html')
# create an endpoint to get the inputs 
@app.route('/GetData')
def getData():
    return render_template('display.html')
# create an endpoint to predict
@app.route('/Predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # fetch the inputs from the form
        city = request.form['city'].strip()
        seller_type = request.form['seller_type']
        bedrooms = int(request.form['bedroom'])
        layout_type = request.form['layout_type']
        property_type = request.form['property_type']
        locality = request.form['locality'].upper()
        area = float(request.form['area'])
        furnish_type = request.form['furnish_type']
        bathroom = int(request.form['bathroom'])
        # loading the encoder for that particular city
        locality_encoder = locality_encoder_dict[city]
        furniture_encoder = furniture_encoder_dict[city]
        layout_encoder = layout_encoder_dict[city]
        property_encoder = property_encoder_dict[city]
        seller_encoder = seller_encoder_dict[city]
        model = model_dict[city] # select the model 
        try:
            # make inputs compatible with our machine learning model 
            locality = locality_encoder.transform([locality])
            furnish_type = furniture_encoder.transform([[furnish_type]])
            layout_type = layout_encoder.transform([[layout_type]])
            property_type = property_encoder.transform([[property_type]])
            seller_type = seller_encoder.transform([[seller_type]])
            # make the prediction 
            preds = model.predict(np.array([
                seller_type,
                bedrooms,
                layout_type,
                property_type,
                locality,
                area,
                furnish_type,
                bathroom
            ],dtype='object').reshape(-1,1).T)
            preds=preds[0]
            preds = np.round(preds)
            return render_template('predict.html',message=f'The prediction is {preds:,}',city=city) # return the predictions
        except:
            return render_template('predict.html',message='Please enter a valid location',city=city) # return error message for wrong location

if __name__ == '__main__':
    app.run(debug=True)