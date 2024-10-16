#!/usr/bin/env python
# coding: utf-8

# In[364]:


# importing all imp libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# In[365]:


# importing data
df = pd.read_csv("Dentistry Dataset.csv",encoding = "utf-8")


# In[366]:


df.head()


# In[367]:


df["Gender"].value_counts()


# In[368]:


# checking data types for all columns
df.dtypes


# In[369]:


df.shape


# In[370]:


# dropping "Sample ID" and "Sl No " because non related
df= df.drop(["Sample ID","Sl No"], axis = 1)


# In[371]:


df.head()


# In[372]:


# to get insight of data
df.describe()


# In[373]:


# To check the null values columns
df.isnull().sum()


# In[374]:


# checking data types
df.dtypes


# In[375]:



from sklearn.preprocessing import LabelEncoder

# Assuming 'df' is your DataFrame and 'Gender' is the column you want to encode

# Initialize the LabelEncoder
le = LabelEncoder()

# Fit and transform the 'Gender' column
df['Gender_encoded'] = le.fit_transform(df['Gender'])

# Display the first few rows to check the result
print(df[['Gender', 'Gender_encoded']].head())


# In[376]:


# drop gender
#df = df.drop("Gender",axis = 1)


# In[377]:


# Fill missing values with column means
df_filled = df.fillna(df.mean)

# Label encode the 'Gender' column
le = LabelEncoder()
df_filled['Gender_encoded'] = le.fit_transform(df_filled['Gender'])


# In[378]:


df_encoded.head()


# In[379]:


# Split independent and dependent variables i.e. X and Y

# dependent veriable
Y = df_encoded["Gender_encoded"]

# independent veriable
X = df_encoded.drop("Gender_encoded",axis=1)


# In[ ]:





# In[380]:


from sklearn.preprocessing import Normalizer

# Initialize the Normalizer
normalizer = Normalizer()

# Apply normalization to the independent variables (X)
X_normalized = normalizer.fit_transform(X)

# Convert the result vack to original data
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Display the first few rows of the normalized_X
print(X_normalized_df.head())


# In[381]:


# Exploratory Data Analysis
corr_matrix = df_encoded.corr()
corr_matrix


# In[382]:


df.dtypes


# In[ ]:





# In[383]:


# handling missing values
df_filled = df.fillna(df.mean)

# Fill missing values with column means
df_filled = df.fillna(df.mean)

# Label encode the 'Gender' column
le = LabelEncoder()
df_filled['Gender_encoded'] = le.fit_transform(df_filled['Gender'])

# Drop the original 'Gender' column
df_encoded = df_filled.drop('Gender', axis=1)


# In[384]:


# Model Building 

   
from sklearn.model_selection import train_test_split

# Split independent & dependent variable i.e X and Y
Y = df_encoded["Gender_encoded"]
X = df_encoded.drop("Gender_encoded",axis=1)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)


# In[385]:


np.float = float    
np.int = int   #module 'numpy' has no attribute 'int'
np.object = object    #module 'numpy' has no attribute 'object'
np.bool = bool 


# In[386]:


# model evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[392]:


#logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, Y_train)
lr.score(X_train, Y_train)
lr.score(X_test, Y_test)


# In[391]:


# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
dt.score(X_train, Y_train)
dt.score(X_test, Y_test)


# In[393]:


# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
rf.score(X_train, Y_train)
rf.score(X_test, Y_test)


# In[394]:


# Random Forest
gbc =  GradientBoostingClassifier()
gbc.fit(X_train, Y_train)
gbc.score(X_train, Y_train)
gbc.score(X_test, Y_test)


# In[ ]:


# result
# all the models are giving gpood score i.e above 75% so we can use all models 

