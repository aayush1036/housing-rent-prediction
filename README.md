# ml-project

This is the github repository created for the Machine Learning project done by:
1. <a href="mailto:aayushmaan1306@gmail.com">Aayushmaan Jain</a>

Which involves creating a web app which is able to take inputs and predict house prices in the following cities:
1. Mumbai
2. Delhi
3. Kolkata
4. Bangalore
5. Hyderabad 
6. Chennai
7. Ahemdabad 
8. Pune

<a href="https://www.kaggle.com/saisaathvik/house-rent-prices-of-metropolitan-cities-in-india">Data Source</a>

<a href="https://rent-prediction-grootis-330517.el.r.appspot.com/">Web App</a>

This project is an end to end implementation of a project where the goal is to predict house rent prices for various different cities<br>
This project uses a machine leraning model in the backend to predict the rent prices in various cities using the inputs given to it<br>
This project uses a XGBoost Regressor model (which turned out to be the best model)<br>
The models were evaluated on the basis of two metrics<br>
<ol>
<li>R<sup>2</sup>score</li>
<li>Mean Absolute Error</li>
</ol>

The models tried out in this project are<br>
<ol>
<li>Linear Regression</li>
<li>Decision Tree Regression</li>
<li>Random Forest Regression</li>
<li>Adaboost Regression</li>
<li>Gradient Boost Regression</li>
<li>XGBoost Regression</li>
</ol>

Running the project on local machine <br>
Step 1 - Clone the project <br>
```
git clone https://github.com/aayush1036/housing-rent-prediction.git
cd housing-rent-prediction
```
Step 2 - Create an environment <br>
```
pip install virtualenv
virualenv YOUR_ENVIRONMENT_NAME
YOUR_ENVIRONMENT_NAME\Scripts\activate
```
Step 3 - Install the dependencies <br>
```
pip install -r requirements.txt
```
Step 4 - Run the code <br>
```
python main.py
```
Optional - Deployment (you should have google cloud sdk installed) <br>
```
gcloud init
# choose the configuration settings and region
gcloud app deploy app.yaml --project YOUR_PROJECT_NAME
```

Directory structure
```
Home - Objects - Encoders(Contains all the label encoders and ordinal encoders for preprocessing)
               - Models (Contains the best selected model for all cities)
     - Results - Contains the results of all the models to aid in model selection
     - static - All the files needed to render flask app
     - templates - All the templates used in the flask app
     - _All_Cities_Cleaned.csv (data source)
     - .gcloudignore - Files to be ignored while deploying the app on GCP
     - .gitignore - Files to be ignored while uploading code on github
     - app.yaml - Specifies the instructions for the runtime while deploying the app on GCP
     - eda.ipynb - Python notebook used while doing Exploratory data analysis 
     - main.py - Python file containing the code for making the flask app
     - models.ipynb - Python notebook used while creating the models 
     - preprocessing.ipynb - Python notebook used while data preprocessing
     - README.md - Readme file for github repository 
     - requirements.txt - For installing the dependencies 
```

App structure<br>
<img src="https://i.ibb.co/wYC5CSy/homepage.png" alt="homepage" border="0">
<br>
Home page

* Contains all the links to navigate through the page at the top
* Contains the information about all cities, where the headings are linked to the page for each city if the user is interested 
* Contains basic overall analysis for the entire data at the bottom 
* Contains contact details in the bottom 
<br>
<img src="https://i.ibb.co/25GCnSz/citypage.png" alt="citypage" border="0"><br>
Pages for a particular city

* Contains all the links to navigate through the page at the top
* Contains a detailed analysis for the houses in that city to help the user in making an informed decision 
* Contains a predict button at the bottom which takes the user to the predict page 
<br>
<img src="https://i.ibb.co/XxMkXGQ/predict.png" alt="predict" border="0"><br>
Predict Page

* Takes the input from the user
* Submit button to display the results
<br>
<img src="https://i.ibb.co/hWFvZMq/result.png" alt="result" border="0"><br>
Results Page

* Displays the result for the particular set of inputs
