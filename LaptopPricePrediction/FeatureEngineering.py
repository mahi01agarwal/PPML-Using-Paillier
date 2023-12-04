# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %%
df = pd.read_csv('./Data/laptop_data.csv')

# %%
df.head()

# %%
df.info()

# %%
df = df.drop('Unnamed: 0',axis = 'columns')

# %%
df.duplicated().sum()

# %%
df.isnull().sum()

# %% [markdown]
# ## Let's first go through Ram and Weight Column

# %%
df["Ram"] = df["Ram"].str.replace("GB","").astype('int32')
df["Weight"] = df["Weight"].str.replace("kg","").astype('float32')

# %%
df['Ram'].value_counts().plot(kind='bar')

# %%
import seaborn as sns
sns.barplot(x=df['Ram'],y=df['Price'])
#plt.xticks(rotation = 'vertical');

# %%
df['Weight'].value_counts()

# %%
sns.distplot(df['Weight'])

# %%
#Seaborn distplot lets you show a histogram with a line on it.
#We use seaborn in combination with matplotlib
import seaborn as sns
sns.distplot(df['Price'])   #Plots the distribution of 'price' data     #Clearly laptoips with less price are most common

# %% [markdown]
# ## Let us now see how Company ,TypeName and OpSys effects Price

# %%
df['Company'].value_counts().plot(kind='bar')

# %%
df['TypeName'].value_counts().plot(kind='bar')

# %%
df['OpSys'].value_counts().plot(kind='bar')    #Clearly Windows10 OpSys is the highest selling

# %%
#let see what is the avg price range for laptops of different comapnies


#A barplot is basically used to aggregate the categorical data according to some methods and by default it’s the mean. It can also be understood as a visualization of the group by action. To use this plot we choose a categorical column for the x-axis and a numerical column for the y-axis, and we see that it creates a plot taking a mean per categorical column.

sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation = 'vertical');

# %%
sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation = 'vertical');

# %%
sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation = 'vertical');

# %%
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

# %%
df['os'] = df['OpSys'].apply(cat_os)

# %%
df.drop('OpSys',axis = 'columns')

# %% [markdown]
# ## Let's now go through Inches column

# %%
df['Inches'].value_counts().plot(kind='bar')  

# %%
sns.distplot(df['Inches'])

# %%
sns.barplot(x=df['Inches'],y=df['Price'])
plt.xticks(rotation = 'vertical');

# %% [markdown]
# ## Now let us work on CPU

# %%
df['Cpu'].value_counts()

# %%
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

# %%
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
df['Cpu Name']

# %%
df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)

# %%
df['Cpu brand'].value_counts().plot(kind ='bar')

# %%
sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation = 'vertical');

# %%
import re
df['ProcessorSpeed'] = df['Cpu'].apply(lambda x: re.findall(r'\d+\.\d+?', x))

# %%
# Function to convert list values to float
def convert_to_float(lst):
    return float(lst[0]) if lst else None

# %%
df['ProcessorSpeed'] = df['ProcessorSpeed'].apply(convert_to_float)

# %%
df = df.drop(['Cpu','Cpu Name'],axis = 'columns')

# %% [markdown]
# ## Now let us work on GPU

# %%
df['Gpu'].value_counts()

# %%
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
            

# %%
import re
df['GpuModel'] = df['Gpu'].apply(lambda x: re.findall(r'\d+', x))

# %%
df['GpuModel']

# %%
df['GpuModel'] = df['GpuModel'].apply(convert_to_float)

# %%
df = df.drop('Gpu',axis = 'columns')

# %%
df.head()

# %% [markdown]
# ## Now let us work on Memory

# %%
df['Memory'].value_counts()

# %%
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)
df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df = df.drop(['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage','Memory'],axis='columns')


# %%
df

# %% [markdown]
# ## Now let us work on Screen Resolution

# %%
df['ScreenResolution']

# %%
df['TouchScreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

# %%
sns.barplot(x=df['TouchScreen'],y=df['Price'])

# %%
df['IPSPanel'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS Panel' in x else 0)

# %%
sns.barplot(x=df['IPSPanel'],y=df['Price'])

# %%
#To get x and y resolution from screen res , if we split it at 'x' sign , we will get our y resolution
new = df['ScreenResolution'].str.split('x',expand=True)

# %%
df['Y_res'] = new[1]

# %%
new[0]

# %%
import re
df['X_res'] = new[0].apply(lambda x: re.findall(r'\d+', x)).apply(lambda x:x[0])

# %%
df['X_res']

# %%
df['X_res']=df['X_res'].astype('int')
df['Y_res']=df['Y_res'].astype('int')

# %%
df = df.drop('ScreenResolution',axis='columns')

# %%
df.head()

# %%
#Let us make a new column pixels per inches
df['PPI']= ((df['X_res']**2)+(df['Y_res'])**2)**0.5/df['Inches']

# %%
df.corr()['Price']

# %%
df = df.drop(['Inches','X_res','Y_res'],axis = 'columns')

# %%
df.head()

# %%
import seaborn as sns
sns.heatmap(df.corr())

# %%
df.corr()['Price']

# %% [markdown]
# ## Working on our target column 'Price'

# %%
sns.distplot(np.log(df['Price']))    #Coz our Price column was skewed we apllied log on it

# %%
df = df.dropna(how ='any')
df.isnull().sum()

# %% [markdown]
# ## Working on Model

# %%
X = df.drop(columns=['Price'])
y = np.log(df['Price'])

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=0)

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression

# %%
# Define the ColumnTransformer for one-hot encoding
col_tnf = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0, 1, 3, 5, 6, 8])
], remainder='passthrough')


# Apply the transformation to X_train and X_test
X_train_transformed = col_tnf.fit_transform(X_train)


import pickle
# Save col_tnf to a file
with open('col_tnf.pkl', 'wb') as file:
    pickle.dump(col_tnf, file)

X_test_transformed = col_tnf.transform(X_test)


# Create and fit the linear regression model
regression_model = LinearRegression()
regression_model.fit(X_train_transformed, y_train)

# Make predictions on the test set
y_pred = regression_model.predict(X_test_transformed)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the evaluation metrics
print('R2 score:', r2)
print('MAE:', mae)


# %%
Coefficients = regression_model.coef_
coef_list = Coefficients.tolist()    
Intercept = regression_model.intercept_

# %%
import json
with open('./Data/Coefficients.json', "w") as file:
    json.dump(coef_list, file)

# %%
with open('./Data/Intercept.json','w') as file:
    json.dump(Intercept,file)

# %% [markdown]
# ## Visualize
# 

# %%
y_test = np.array(y_test)
plt.plot(y_pred,label = 'Actual')
plt.plot(y_test,label = 'Predicted')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend(loc='best')

# %%



