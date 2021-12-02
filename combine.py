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
# columns list 
cols = ['SELLER TYPE','BEDROOM','LAYOUT TYPE','PROPERTY TYPE','LOCALITY','PRICE','AREA','FURNISH TYPE','BATHROOM']
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

# Reading credentials 
with open('credentials.txt', 'r') as f:
    PASSWORD = f.read().strip()

# Initializing dictionaries 
clean_df_dict = {}
preprocessed_df_dict = {}
X_train_dict = {}
y_train_dict = {}

# connecting to mysql 
conn = mysql.connector.connect(
    host='34.93.147.30',
    port=3306,
    user='root',
    password=PASSWORD,
    database='CLEAN',
    auth_plugin='mysql_native_password'
)
# reading the data and storing it in dictionaries 
df_dict = dict(zip(CITIES, [pd.read_sql(f'SELECT * FROM {city}', con=conn) for city in CITIES]))
# closing the connection 
conn.close()

# preprocessing the data 
def preprocess(df_dict:dict)-> None:
    """Preprocesses the data according to the following steps: 
    1) Remove duplicates
    2) Remove outliers (i.e below Q1-1.5*IQR or above Q3+1.5*IQR) 
    3) Label Encode locality - as the order does not matter while encoding locality
    4) Ordinal Encode property type - as the order does matter while encoding property type
    5) Ordinal Encode seller type - as the order does matter while encoding seller type 
    6) Ordinal Encode layout type - as the order does matter while encoding layout type 
    7) Ordinal Encode furnish type - as the order does matter while encoding furnish type
    
    Keyword arguments:
    df_dict(dict) -- The dictionary containing the dataframes for preprocessing
    Return: None
    """
    for city, df in df_dict.items():
        # Rename columns for uniformity
        df.columns = cols 
        # Dropping the duplicates (if any)
        df.drop_duplicates(inplace=True)
        # Converting locality to upper case 
        df['LOCALITY'] = df['LOCALITY'].str.upper()
        # Removing outliers 
        desc = df['PRICE'].describe()
        # Getting IQR
        iqr = desc.loc['75%'] - desc.loc['25%']
        # Calculating lower limit as Q1-(1.5*IQR)
        lower_limit = desc.loc['25%'] - 1.5*iqr
        # Calculating upper limit as Q3+(1.5*IQR)
        upper_limit = desc.loc['75%'] + 1.5*iqr
        # Subsetting the data so as to remove outliers 
        df = df[(df['PRICE']>=lower_limit)&(df['PRICE']<=upper_limit)]
        # storing the data in clean_df_dict for EDA later on
        clean_df_dict[city] = df.copy()
        # Building the categories for ordinal encoder 
        cat_furnish = [['Unfurnished','Semi-Furnished','Furnished']]
        cat_seller = [df.groupby(by=['SELLER TYPE'])['PRICE'].mean().sort_values(ascending=True).index.values.tolist()]
        cat_layout_type = [df.groupby(by=['LAYOUT TYPE'])['PRICE'].mean().sort_values(ascending=True).index.values.tolist()]
        cat_property_type = [df.groupby(by=['PROPERTY TYPE'])['PRICE'].mean().sort_values(ascending=True).index.values.tolist()]
        # creating ordinal encoders
        furnish_type_encoder = OrdinalEncoder(categories=cat_furnish)
        seller_type_encoder = OrdinalEncoder(categories=cat_seller)
        layout_type_encoder = OrdinalEncoder(categories=cat_layout_type)
        property_type_encoder = OrdinalEncoder(categories=cat_property_type)
        # creating label encoders 
        locality_encoder = LabelEncoder()
        # Applying the transformations 
        df['LOCALITY'] = locality_encoder.fit_transform(df['LOCALITY'])
        df['FURNISH TYPE'] = furnish_type_encoder.fit_transform(df[['FURNISH TYPE']])
        df['SELLER TYPE'] = seller_type_encoder.fit_transform(df[['SELLER TYPE']])
        df['PROPERTY TYPE'] = property_type_encoder.fit_transform(df[['PROPERTY TYPE']])
        df['LAYOUT TYPE'] = layout_type_encoder.fit_transform(df[['LAYOUT TYPE']])
        # checking if the path exists
        paths = [FURNITURE_ENCODER_PATH, LAYOUT_ENCODER_PATH, LOCALITY_ENCODER_PATH, SELLER_ENCODER_PATH, PROPERTY_ENCODER_PATH]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
        # saving the encoders 
        joblib.dump(furnish_type_encoder, os.path.join(FURNITURE_ENCODER_PATH, f'{city}_furnish_type_encoder.pkl'))
        joblib.dump(seller_type_encoder, os.path.join(SELLER_ENCODER_PATH, f'{city}_seller_type_encoder.pkl'))
        joblib.dump(layout_type_encoder, os.path.join(LAYOUT_ENCODER_PATH, f'{city}_layout_type_encoder.pkl'))
        joblib.dump(property_type_encoder, os.path.join(PROPERTY_ENCODER_PATH, f'{city}_property_type_encoder.pkl')) 
        joblib.dump(locality_encoder, os.path.join(LOCALITY_ENCODER_PATH, f'{city}_locality_encoder.pkl'))
        # save the dataframe in preprocessed_df_dict
        preprocessed_df_dict[city] = df
preprocess(df_dict)

# EDA 
for city, df in clean_df_dict.items():
    df['CITY'] = city
    df['AFFORDABILITY'] = df['AREA']/df['PRICE']

combined = pd.concat([clean_df_dict[city] for city in CITIES])

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
    if not os.path.exists(os.path.join('outputs',city)): #check if the path exists 
        os.makedirs(os.path.join('outputs',city)) #if not, make the path 
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
    affordable = df.groupby(by=['LOCALITY'])['AFFORDABILITY'].mean() #calculate the mean price for each area
    most_affordable = affordable.sort_values(ascending=True)[:10] #sort them in ascending order for most affordable 
    least_affordable = affordable.sort_values(ascending=False)[:10] #sort them in descending order for least affordable
    fig, ax = plt.subplots(figsize=MULTICOL_FIGSIZE,nrows=1,ncols=2) #create a figure of 1 row and 2 cols 
    sns.barplot(x=most_affordable.index, y=most_affordable,ax=ax[0]) #plot the most affordable areas on 1st column 
    ax[0].set_title(f'MOST AFFORDABLE LOCALITIES IN {city}') #set the title 
    ax[0].set_xlabel('LOCALITY') #set the xlabel 
    ax[0].set_ylabel('AVERAGE PRICE PER SQUARE FEET') #set the ylabel 
    ax[0].tick_params(axis='x',labelrotation=90) #rotate the labels on x axis by 90 degrees for readibility
    sns.barplot(x=least_affordable.index, y=least_affordable, ax=ax[1]) #plot the least affordable areas on 2nd column 
    ax[1].set_title(f'LEAST AFFORDABLE LOCALITIES IN {city}') #set the title 
    ax[1].set_xlabel('LOCALITY') #set the xlabel 
    ax[1].set_ylabel('AVERAGE PRICE PER SQUARE FEET') #set the ylabel 
    ax[1].tick_params(axis='x',labelrotation=90) #rotate the labels on x axis by 90 degrees for readibility
    plt.tight_layout() #tight layout to prevent overlapping 
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