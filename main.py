from flask import Flask, render_template,request
import numpy as np
from utils import loadModels,loadEncoders, getDataFromForm, createConnection, transformData
# CONSTANTS 
CITIES = ['AHEMDABAD','BANGALORE','CHENNAI','DELHI','HYDERABAD','KOLKATA','MUMBAI','PUNE']
model_dict = loadModels()
encoders = loadEncoders()
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
        city, values = getDataFromForm(request=request)
        try:
            # make inputs compatible with our machine learning model 
            data = transformData(city=city, data=values, encoders=encoders)
            model = model_dict[city]
            # make the prediction 
            preds = model.predict(data)
            preds=preds[0]
            preds = np.round(preds)
            message = f'The prediction is Rs {preds:,} for {city}'
            return render_template('display.html',statusCode='Success',message=message) # return the predictions
        except:
            message = 'The location that you have entered is not registered in our database'
            return render_template('display.html',statusCode = 'Failure',message=message) # return error message for wrong location
@app.route('/GetCorrections')
def getCorrections():
    return render_template('contribute.html')
@app.route('/Contribute',methods=['GET','POST'])
def contribute():
    if request.method == 'POST':
        city, values = getDataFromForm(request=request, contribute=True)
        try:
            # make a connection to MySQL
            conn = createConnection()
            # create a cursor to execute queries in the connection 
            cursor = conn.cursor()
            # frame the SQL query 
            query = f"""
            INSERT INTO {city} (SELLER_TYPE, BEDROOM, LAYOUT, PROPERTY_TYPE, LOCALITY, PRICE, AREA, FURNISH_TYPE, BATHROOM)
            VALUES {tuple(values)};
            """
            # execute the query 
            cursor.execute(query)
            # commit the query to the connection
            conn.commit()
            # close the connection
            conn.close()
            return render_template('thanks.html', status='success') # return the predictions
        except:
            return render_template('thanks.html', status='failure') # return error message for wrong location
    
if __name__ == '__main__':
    app.run(debug=True)