from flask import Flask, render_template,request
import numpy as np
from utils import loadModels,loadEncoders, getDataFromForm, createConnection, transformData

model_dict = loadModels()
encoders = loadEncoders()
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/Ahemdabad')
def ahemdabad():
    return render_template('ahemdabad.html')
@app.route('/Bangalore')
def bangalore():
    return render_template('bangalore.html')
@app.route('/Chennai')
def chennai():
    return render_template('chennai.html')
@app.route('/Delhi')
def delhi():
    return render_template('delhi.html')
@app.route('/Hyderabad')
def hyderabad():
    return render_template('hyderabad.html')
@app.route('/Kolkata')
def kolkata():
    return render_template('kolkata.html')
@app.route('/Mumbai')
def mumbai():
    return render_template('mumbai.html')
@app.route('/Pune')
def pune():
    return render_template('pune.html')
@app.route('/GetData')
def getData():
    return render_template('display.html')
@app.route('/Predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        city, values = getDataFromForm(request=request)
        try:
            data = transformData(city=city, data=values, encoders=encoders)
            model = model_dict[city]
            preds = np.round(model.predict(data)[0])
            message = f'The prediction is Rs {preds:,} for {city}'
            return render_template('display.html',statusCode='Success',message=message)
        except:
            message = 'The location that you have entered is not registered in our database'
            return render_template('display.html',statusCode = 'Failure',message=message)
@app.route('/GetCorrections')
def getCorrections():
    return render_template('contribute.html')
@app.route('/Contribute',methods=['GET','POST'])
def contribute():
    if request.method == 'POST':
        city, values = getDataFromForm(request=request, contribute=True)
        try:
            conn = createConnection()
            cursor = conn.cursor()
            query = f"""
            INSERT INTO {city} (SELLER_TYPE, BEDROOM, LAYOUT, PROPERTY_TYPE, LOCALITY, PRICE, AREA, FURNISH_TYPE, BATHROOM)
            VALUES {tuple(values)};
            """
            cursor.execute(query)
            conn.commit()
            conn.close()
            return render_template('thanks.html', status='success')
        except:
            return render_template('thanks.html', status='failure')
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')