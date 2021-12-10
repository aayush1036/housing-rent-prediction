# IMPORTS  
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import mysql.connector
import os 
import warnings
from sklearn.metrics import r2_score
import datetime as dt
import json
plt.style.use('seaborn')
warnings.simplefilter(action='ignore')

# CONSTANTS 

# filepaths for saving and retrieving the encoders 
LOCALITY_ENCODER_PATH = 'Objects/Encoders/LabelEncoder/'
LAYOUT_ENCODER_PATH = 'Objects/Encoders/OrdinalEncoder/layout_type/'
PROPERTY_ENCODER_PATH = 'Objects/Encoders/OrdinalEncoder/property_type/'
SELLER_ENCODER_PATH = 'Objects/Encoders/OrdinalEncoder/seller_type/'
FURNITURE_ENCODER_PATH = 'Objects/Encoders/OrdinalEncoder/furniture_encoders/'
# city list 
CITIES = ['AHEMDABAD','BANGALORE','CHENNAI','DELHI','HYDERABAD','KOLKATA','MUMBAI','PUNE']
# for plotting 
PIE_NROWS = 2
PIE_NCOLS = 2
NUMERICAL_NROWS = 2
NUMERICAL_NCOLS = 2
REGULAR_FIGSIZE = (16,6)
PIE_FIGSIZE=(20,15)
MULTICOL_FIGSIZE=(16,8)
# for saving plots 
AFFORDABILITY_PLOT_DESTINATION = 'static/Affordability/'
OTHER_PLOT_DESTINATION = 'static/Other/'
NUMERICAL_PLOT_DESTINATION = 'static/Numerical/'
SPACIOUS_PLOT_DESTINATION = 'static/Spacious/'
OVERALL_NUMERICAL_ANALYSIS_PATH = 'static/Overall'
# for saving models 
RESULTS_SAVE_PATH = 'Results/'
MODEL_SAVE_PATH = 'Objects/Models/'
EVALUATION_FIGSIZE = (16,6)
EVALUATION_NROWS = 1
EVALUATION_NCOLS = 2
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Initializing dictionaries 
clean_df_dict = {}
preprocessed_df_dict = {}
X_train_dict = {}
y_train_dict = {}

# getting the config file 
with open('config.json', 'r') as f:
    config = json.load(f)

# connecting to mysql 
conn = mysql.connector.connect(
    host=config.get('host'),
    port=config.get('port'),
    user=config.get('user'),
    password=config.get('password'),
    database=config.get('database'),
    auth_plugin=config.get('auth_plugin')
)
# reading the data and storing it in dictionaries 
df_dict = dict(zip(CITIES, [pd.read_sql(f'SELECT * FROM {city}', con=conn) for city in CITIES]))
# closing the connection 
conn.close()

# for storing encoders while preprocessing 
locality_encoder_dict = {}
furnish_type_encoder_dict = {}
seller_type_encoder_dict = {}
layout_type_encoder_dict = {}
property_type_encoder_dict = {}

# for storing encoders while preprocessing 
locality_encoder_dict = {}
furnish_type_encoder_dict = {}
seller_type_encoder_dict = {}
layout_type_encoder_dict = {}
property_type_encoder_dict = {}
def preprocess(df_dict):
    """
    Cleans and preprocesses the data for future use 
    Cleaning:
        - Removes duplicate rows (if they exist)
        - Removes the city column as the dataframes are already separated 
        - Removes outliers on the basis of price column using lower limit as (Q1-1.5*IQR) and 
        upper limit as (Q3+1.5*IQR)
    Preprocessing:
        - Uses label encoding for locality column as order does not matter in locality
        - Uses ordianal encoding for furnish_type, layout_type, seller_type and property_type as the 
        order matters in these columns for predicting the rent prices
    Args:
        df_dict - the dictionary containing cities as keys and the dataframe corresponding to them as values 
    Returns:
        None 
    """
    for city, df in df_dict.items():
        # Cleaning the data 
        # Renaming the columns
        df.drop(['ID'],axis=1, inplace=True)
        df.columns = ['SELLER TYPE','BEDROOM','LAYOUT TYPE','PROPERTY TYPE','LOCALITY','PRICE','AREA','FURNISH TYPE','BATHROOM']
        # Dropping duplicate rows (if they exist)
        df.drop_duplicates(inplace=True)
        # Removing outliers 
        desc = df['PRICE'].describe()
        q1 = desc.loc['25%']
        q3 = desc.loc['75%']
        iqr = q3-q1 
        lower_lim = q1-(1.5*iqr)
        upper_lim = q3+(1.5*iqr)
        df = df[(df['PRICE']>=lower_lim)&(df['PRICE']<=upper_lim)]
        # Renaming the columns 
        cols = ['SELLER TYPE','BEDROOM','LAYOUT TYPE','PROPERTY TYPE','LOCALITY','PRICE','AREA','FURNISH TYPE','BATHROOM']
        df.columns = cols
        # since the data is cleaned now, we can store it in clean_dict 
        clean_df = df.copy()
        clean_df_dict[city] = clean_df
        # Preprocessing the data 
        locality_encoder = LabelEncoder()
        df['LOCALITY'] = locality_encoder.fit_transform(df['LOCALITY'])
        locality_encoder_dict[city] = locality_encoder
        if not os.path.exists(LOCALITY_ENCODER_PATH):
            joblib.dump(locality_encoder)
        
        ordinal_encoder_cols = ['SELLER TYPE','LAYOUT TYPE','PROPERTY TYPE','FURNISH TYPE']        
        ord_enc_dict = {
            'SELLER TYPE':seller_type_encoder_dict,
            'LAYOUT TYPE':layout_type_encoder_dict,
            'PROPERTY TYPE':property_type_encoder_dict,
            'FURNISH TYPE':furnish_type_encoder_dict
        }
        for col in ordinal_encoder_cols:
            cat = [df.groupby(by=[col])['PRICE'].mean().sort_values(ascending=True).index]
            col_encoder = OrdinalEncoder(categories=cat)
            df[col] = col_encoder.fit_transform(df[[col]])
            ord_enc_dict[col][city] = col_encoder
        preprocessed_df_dict[city] = df
        paths = {
            'SELLER TYPE': os.path.join(SELLER_ENCODER_PATH, f'{city}_seller_type_encoder.pkl'),
            'LAYOUT TYPE': os.path.join(LAYOUT_ENCODER_PATH,f'{city}_layout_type_encoder.pkl'),
            'PROPERTY TYPE': os.path.join(PROPERTY_ENCODER_PATH,f'{city}_property_type_encoder.pkl'),
            'FURNISH TYPE': os.path.join(FURNITURE_ENCODER_PATH, f'{city}_furnish_type_encoder.pkl')
        }

        if not os.path.exists(FURNITURE_ENCODER_PATH): #check if the desired file path exists
            os.makedirs(FURNITURE_ENCODER_PATH) #if not then make one 

        if not os.path.exists(SELLER_ENCODER_PATH):
            os.makedirs(SELLER_ENCODER_PATH)
        
        if not os.path.exists(LAYOUT_ENCODER_PATH):
            os.makedirs(LAYOUT_ENCODER_PATH)
        
        if not os.path.exists(PROPERTY_ENCODER_PATH):
            os.makedirs(PROPERTY_ENCODER_PATH)
     
        for col in ordinal_encoder_cols:
            joblib.dump(ord_enc_dict[col][city], paths[col])
preprocess(df_dict)

# EDA 
for city, df in clean_df_dict.items():
    df['CITY'] = city
    df['AFFORDABILITY'] = df['AREA']/df['PRICE']

combined = pd.concat([clean_df_dict[city] for city in CITIES])
# Overall Analysis (for home page)
# Preferred Cities
fig, ax = plt.subplots(figsize=REGULAR_FIGSIZE)
sns.countplot(x=combined['CITY'],ax=ax)
ax.set_xlabel('CITY')
ax.set_ylabel('NUMBER OF HOUSES')
ax.set_title('NUMBER OF HOUSES IN EACH CITY')
if not os.path.exists(OVERALL_NUMERICAL_ANALYSIS_PATH):
    os.makedirs(OVERALL_NUMERICAL_ANALYSIS_PATH)
plt.savefig(os.path.join(OVERALL_NUMERICAL_ANALYSIS_PATH, 'n_houses.png'))

# Other columns 
overall_numerical_cols = ['PRICE','AREA','AFFORDABILITY']
for col in overall_numerical_cols:
    fig, ax = plt.subplots(figsize=REGULAR_FIGSIZE)
    mean_df = combined.groupby(by=['CITY'])[col].mean()
    mean_df.sort_values(inplace=True,ascending=False)
    sns.barplot(x=mean_df.index, y=mean_df, ax=ax)
    ax.set_xlabel('CITY')
    ax.set_ylabel(f'AVERAGE {col}')
    ax.set_title(f'AVERAGE {col} IN EACH CITY')
    plt.savefig(os.path.join(OVERALL_NUMERICAL_ANALYSIS_PATH, f'{col}.png'))

# Number of houses in each city
fig, ax = plt.subplots(figsize=REGULAR_FIGSIZE)
sns.countplot(x=combined['CITY'],ax=ax,order=combined['CITY'].value_counts().index)
ax.set_title('NUMBER OF HOUSES IN EACH CITY')
ax.set_xlabel('CITY')
ax.set_ylabel('NUMBER OF HOUSES')
if not os.path.exists(OVERALL_NUMERICAL_ANALYSIS_PATH):
    os.makedirs(OVERALL_NUMERICAL_ANALYSIS_PATH)
plt.savefig(os.path.join(OVERALL_NUMERICAL_ANALYSIS_PATH, 'n_houses.png'))

# Overall analysis
overall_numerical_cols = ['PRICE','AREA','AFFORDABILITY']
for col in overall_numerical_cols:
    fig, ax = plt.subplots(figsize=REGULAR_FIGSIZE)
    mean_df = combined.groupby(by=['CITY'])[col].mean()
    mean_df.sort_values(inplace=True,ascending=False)
    sns.barplot(x=mean_df.index, y=mean_df, ax=ax)
    ax.set_xlabel('CITY')
    ax.set_ylabel(f'AVERAGE {col}')
    ax.set_title(f'AVERAGE {col} IN EACH CITY')
    plt.savefig(os.path.join(OVERALL_NUMERICAL_ANALYSIS_PATH, f'{col}.png'))

# Other analysis
pie_cols = np.array(['SELLER TYPE','LAYOUT TYPE','PROPERTY TYPE','FURNISH TYPE']).reshape(PIE_NROWS,PIE_NCOLS)
for city, df in clean_df_dict.items():
    fig, ax = plt.subplots(figsize=PIE_FIGSIZE,nrows=PIE_NROWS, ncols=PIE_NCOLS) #create a fig with 2 rows and 2 cols 
    for i in range(PIE_NROWS): #loop through the rows 
        for j in range(PIE_NROWS): #loop through columns 
            ax[i,j].pie(x=df[pie_cols[i,j]].value_counts()) #plot the pie chart 
            text = pd.DataFrame(df[pie_cols[i,j]].value_counts().apply(lambda x: f'{np.round((x/df.shape[0])*100,2)}%')) 
            # create the text to display on pie chart 
            text.index = text.index.str.upper() #convert text to upper case 
            text = text.to_string() #convert text to string 
            ax[i,j].text(1,0,text) #display text on pie chart
            ax[i,j].set_title(f'{pie_cols[i,j]} IN {city}') #set the title 
    if not os.path.exists(OTHER_PLOT_DESTINATION): #check if the path exists 
        os.makedirs(OTHER_PLOT_DESTINATION) #if not then create the path 
    plt.savefig(os.path.join(OTHER_PLOT_DESTINATION, f'{city}.png')) #save the figure 

# Numerical analysis 
# create an array for analyzing numerical columns 
numerical_cols = np.array([
    ['PRICE','AREA'],
    ['BEDROOM','BATHROOM']
]).reshape(NUMERICAL_NROWS,NUMERICAL_NCOLS)

for city, df in clean_df_dict.items():
    fig, ax = plt.subplots(figsize=MULTICOL_FIGSIZE,nrows=2,ncols=2) #create a matplotlib figure 
    for i in range(NUMERICAL_NROWS): #loop through the rows 
        for j in range(NUMERICAL_NCOLS): #loop through the columns 
            if i==0: # if it is the 1st row 
                sns.histplot(df[numerical_cols[i,j]],ax=ax[i,j],kde=True) #plot the distribution of column of ith row and jth column
                ax[i,j].set_title(f'DISTRIBUTION OF {numerical_cols[i,j]} IN {city}') # set the title 
                ax[i,j].set_xlabel(numerical_cols[i,j]) #set the xlabel 
                ax[i,j].set_ylabel('NUMBER OF HOUSES') #set the ylabel
            if i==1: #if it is the second row 
                sns.countplot(x=df[numerical_cols[i,j]],ax=ax[i,j]) #plot the countplot of column of ith row and jth column 
                ax[i,j].set_title(f'NUMBER OF {numerical_cols[i,j]} IN THE HOUSES IN {city}') #set the title 
                ax[i,j].set_xlabel(numerical_cols[i,j]) #set the xlabel
                ax[i,j].set_ylabel('NUMBER OF HOUSES') #set the ylabel
    plt.tight_layout() #apply tight layout to prevent overlap of columns  
    if not os.path.exists(NUMERICAL_PLOT_DESTINATION): #check if the path exists 
        os.makedirs(NUMERICAL_PLOT_DESTINATION) #if not then make the path 
    plt.savefig(os.path.join(NUMERICAL_PLOT_DESTINATION, f'{city}.png')) #save the figure at the path 

# Affordability analysis
for city, df in clean_df_dict.items(): 
    affordable = df.groupby(by=['LOCALITY'])['AFFORDABILITY'].mean() #calculate mean area for each locality 
    most_affordable = affordable.sort_values(ascending=False)[:10] #sort in ascending order for most spacious 
    least_affordable = affordable.sort_values(ascending=True)[:10] #sort in descending order for least spacious 
    fig, ax = plt.subplots(figsize=MULTICOL_FIGSIZE,nrows=1,ncols=2) #create figure with 1 row and 2 cols 
    sns.barplot(x=least_affordable.index, y=least_affordable, ax=ax[0], order=least_affordable.index[::-1]) #plot least spacious on 1st col 
    ax[0].set_title(f'LEAST AFFORDABLE LOCALITIES IN {city}') #set title 
    ax[0].set_xlabel('LOCALITY') #set xlabel 
    ax[0].set_ylabel('AVERAGE AFFORDABLITY') #set ylabel
    ax[0].tick_params(axis='x',labelrotation=90) #rotate the labels on x axis by 90 degrees for readibility
    sns.barplot(x=most_affordable.index, y=most_affordable,ax=ax[1],order=most_affordable.index[::-1])#plot least affordable localities in 2nd column 
    ax[1].set_title(f'MOST AFFORDABLE LOCALITIES IN {city}') 
    ax[1].set_xlabel('LOCALITY') #set xlabel 
    ax[1].set_ylabel('AVERAGE AFFORDABLITY') #set ylabel
    ax[1].tick_params(axis='x',labelrotation=90) #rotate the labels on x axis by 90 degrees for readibility
    plt.tight_layout() #apply tight layout for no overlap  
    if not os.path.exists(AFFORDABILITY_PLOT_DESTINATION): #check if the path exists
        os.makedirs(AFFORDABILITY_PLOT_DESTINATION) #if not, make the path 
    plt.savefig(os.path.join(AFFORDABILITY_PLOT_DESTINATION, f'{city}.png')) #save the figure 

# Area analysis 
for city, df in clean_df_dict.items():
    spacious = df.groupby(by=['LOCALITY'])['AREA'].mean() #calculate mean area for each locality 
    most_spacious = spacious.sort_values(ascending=False)[:10] #sort in ascending order for most spacious 
    least_spacious = spacious.sort_values(ascending=True)[:10] #sort in descending order for least spacious 
    fig, ax = plt.subplots(figsize=MULTICOL_FIGSIZE,nrows=1,ncols=2) #create figure with 1 row and 2 cols 
    sns.barplot(x=least_spacious.index, y=least_spacious, ax=ax[0]) #plot least spacious on 1st col 
    ax[0].set_title(f'LEAST SPACIOUS LOCALITIES IN {city}') #set title 
    ax[0].set_xlabel('LOCALITY') #set xlabel 
    ax[0].set_ylabel('AVERAGE AREA IN SQUARE FEET') #set ylabel
    ax[0].tick_params(axis='x',labelrotation=90) #rotate the labels on x axis by 90 degrees for readibility
    sns.barplot(x=most_spacious.index, y=most_spacious,ax=ax[1])#plot least affordable localities in 2nd column 
    ax[1].set_title(f'MOST SPACIOUS LOCALITIES IN {city}') 
    ax[1].set_xlabel('LOCALITY') #set xlabel 
    ax[1].set_ylabel('AVERAGE AREA IN SQUARE FEET') #set ylabel
    ax[1].tick_params(axis='x',labelrotation=90) #rotate the labels on x axis by 90 degrees for readibility
    plt.tight_layout() #apply tight layout for no overlap 
    if not os.path.exists(SPACIOUS_PLOT_DESTINATION): #check if path exists
        os.makedirs(SPACIOUS_PLOT_DESTINATION) #if not then create the path 
    plt.savefig(os.path.join(SPACIOUS_PLOT_DESTINATION, f'{city}.png')) #save the figure 

train_r2_dict = {}
test_r2_dict = {}
# Model building 
for city, df in preprocessed_df_dict.items():
    # Grab X and y from the data 
    X = df.drop(['PRICE'], axis=1)
    y = df['PRICE']
    # perform train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # Since XGBoost is the best model (from models.ipynb), we will retrain a XGBRegressor model every time 
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    # calculate r2_score for evaulation and write them to a file for evaluationn
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_r2 = r2_score(y_true=y_train, y_pred=train_preds)
    test_r2 = r2_score(y_true=y_test, y_pred=test_preds)
    train_r2_dict[city] = train_r2
    test_r2_dict[city] = test_r2
    # Save the model
    joblib.dump(model, os.path.join(MODEL_SAVE_PATH, f'{city}_model.pkl'))
with open('model_results.txt', 'a+') as f:
    f.write(f'Ran on {dt.datetime.now()}\n')
    f.write('Train r2\n')
    f.write(str(train_r2_dict))
    f.write('\n')
    f.write('Test r2\n')
    f.write(str(test_r2_dict))
    f.write('\n')

print(f'Successfully executed on {dt.datetime.now()}')