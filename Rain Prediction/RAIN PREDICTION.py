#!/usr/bin/env python
# coding: utf-8

# # Q1. To predict if it will rain tomorrow in XYZ country using suitable ML approach.

# ## 1. Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Import Dataset

# In[2]:


df=pd.read_csv('weatherAUS.csv')


# In[3]:


#Preview of the dataset
df.head()


# ## 3. Exploratory Data Analysis

# In[4]:


#View dimensions of dataset 
df.shape


# In[5]:


#View statistical properties of dataset
df.describe()


# In[6]:


#View summary of dataset 
df.info()


# In[7]:


#View unique number of values in each column
df.nunique()


# In[8]:


#View number of null values in each column
df.isnull().sum()


# ## 4. Visualizing Null Values 

# In[9]:


#Plotting a heatmap to analyse the null values in the Dataset
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[10]:


#Checking percentage of missing data in every column
(df.isnull().sum()/len(df))*100


# In[11]:


# find categorical variables
categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :', categorical)


# In[12]:


#Filling the missing values for continuous variables with mean
df['MinTemp']=df['MinTemp'].fillna(df['MinTemp'].mean())
df['MaxTemp']=df['MinTemp'].fillna(df['MaxTemp'].mean())
df['Rainfall']=df['Rainfall'].fillna(df['Rainfall'].mean())
df['Evaporation']=df['Evaporation'].fillna(df['Evaporation'].mean())
df['Sunshine']=df['Sunshine'].fillna(df['Sunshine'].mean())
df['WindGustSpeed']=df['WindGustSpeed'].fillna(df['WindGustSpeed'].mean())
df['WindSpeed9am']=df['WindSpeed9am'].fillna(df['WindSpeed9am'].mean())
df['WindSpeed3pm']=df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].mean())
df['Humidity9am']=df['Humidity9am'].fillna(df['Humidity9am'].mean())
df['Humidity3pm']=df['Humidity3pm'].fillna(df['Humidity3pm'].mean())
df['Pressure9am']=df['Pressure9am'].fillna(df['Pressure9am'].mean())
df['Pressure3pm']=df['Pressure3pm'].fillna(df['Pressure3pm'].mean())
df['Cloud9am']=df['Cloud9am'].fillna(df['Cloud9am'].mean())
df['Cloud3pm']=df['Cloud3pm'].fillna(df['Cloud3pm'].mean())
df['Temp9am']=df['Temp9am'].fillna(df['Temp9am'].mean())
df['Temp3pm']=df['Temp3pm'].fillna(df['Temp3pm'].mean())


# In[13]:


#Filling the missing values for categorical variables with mode
df['RainToday']=df['RainToday'].fillna(df['RainToday'].mode()[0])
df['RainTomorrow']=df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])


# In[14]:


#Filling the missing values for categorical variables with mode
df['WindDir9am'] = df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
df['WindGustDir'] = df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
df['WindDir3pm'] = df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])


# In[15]:


#Plotting a heatmap to check if all the null values are filled
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ## 5. Chaniging the values Rain today and Rain Tomorrow

# In[16]:


#Changing yes and no to 1 and 0 in some columns
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})

print(df.RainToday)
print(df.RainTomorrow)


# In[17]:


df.head()


# In[18]:


#Dropping date column as it is not necessary for the prediction
df=df.iloc[:,1:]
df.head()


# ## 6. Visualizing the Data

# In[19]:


plt.figure(figsize=(13,11))
ax = sns.heatmap(df.corr(), square=True, annot=True, fmt='.2f')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)          
plt.show()


# ### Count of rain today and tomorrow

# In[20]:


fig, ax =plt.subplots(1,2)
plt.figure(figsize=(20,20))
sns.countplot(data=df,x='RainToday',ax=ax[0])
sns.countplot(data=df,x='RainTomorrow',ax=ax[1])


# ### Direction of wind at 9 am, 3 pm

# In[21]:


fig, ax =plt.subplots(3,1)
plt.figure(figsize=(10,10))

sns.countplot(data=df,x='WindDir9am',ax=ax[0])
sns.countplot(data=df,x='WindDir3pm',ax=ax[1])
sns.countplot(data=df,x='WindGustDir',ax=ax[2])
fig.tight_layout()


# #### Observations:
# - At 9 am, it is highest for direction N.
# - At 3 pm, it is highest for direction SE.

# ### Boxplots for Humidity and Pressure at 3pm and 9am

# In[22]:


fig, ax =plt.subplots(2,1)
plt.figure(figsize=(10,10))
sns.boxplot(x=df['Humidity3pm'],color='c',ax=ax[0])
sns.boxplot(x=df['Humidity9am'],color='c',ax=ax[1])
fig.tight_layout()


# In[23]:


fig, ax =plt.subplots(2,1)
plt.figure(figsize=(10,10))
sns.boxplot(x=df['Pressure3pm'],color='c',ax=ax[0])
sns.boxplot(x=df['Pressure9am'],color='c',ax=ax[1])
fig.tight_layout()


# ### ViolinPlots for RainToday vs MaxTemp and MinTemp

# In[24]:


sns.violinplot(x='RainToday',y='MaxTemp',data=df,hue='RainTomorrow')


# In[25]:


sns.violinplot(x='RainToday',y='MinTemp',data=df,hue='RainTomorrow')


# ## 7. Encoding the categorical variables

# In[26]:


#Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])
df['WindDir9am'] = le.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = le.fit_transform(df['WindDir3pm'])
df['WindGustDir'] = le.fit_transform(df['WindGustDir'])
df.head()


# In[27]:


#Removing the outliers
from scipy import stats
print('Shape of DataFrame Before Removing Outliers', df.shape )
df=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
print('Shape of DataFrame After Removing Outliers', df.shape )


# In[28]:


#Dropping highly correlated columns
#df=df.drop(['Temp3pm','Temp9am','Humidity9am'],axis=1)
#df.columns


# In[29]:


#importing all necessary libraries for training the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# ## 8. Feature Scaling

# In[30]:


#Scaling the data to fit the modedl
scale = MinMaxScaler()


# In[31]:


scaled_df = scale.fit_transform(df.drop('RainTomorrow',axis=1))
scaled_df


# In[32]:


#creating the dataframe with the scaled array data
scaled_df = pd.DataFrame(scaled_df,columns=df.columns[:-1])
scaled_df.head()


# ## 9. Model Training and Predicting results

# In[33]:


#Train test split
X=scaled_df.iloc[:,:-1] 
y=scaled_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[34]:


#Function that fits the model and predicts results
def model_prediction(model, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('Confusion Matrix: \n',confusion_matrix(y_test, preds))
    print('Accuracy of ', title, ':', round(accuracy_score(y_test, preds), 5))


# ## Check accuracy score for Logistic Regression and Random Forest Classifier

# In[35]:


# instantiate the model
lr = LogisticRegression(max_iter=1000)
#passing the model into the fuction to make predictions
model_prediction(lr, "Logistic Regression")


# In[36]:


# instantiate the model with Random Forest
rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model_prediction(rf, "Random Forest")


# ## Check for overfitting and underfitting
# ### Compare the train-set and test-set accuracy for both models

# In[37]:


y_pred_train = lr.predict(X_train)
print('Training-set accuracy score for Logistic Regression mode : {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[38]:


y_pred_train_rf = rf.predict(X_train)
print('Training-set accuracy score for Random Forest model:: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_rf)))


# In[39]:


print('Training set score for Logistic Regression model: {:.4f}'.format(lr.score(X_train, y_train)))
print('Test set score for Logistic Regression model: {:.4f}'.format(lr.score(X_test, y_test)))


# In[40]:


print('Training set score for Random Forest model: {:.4f}'.format(rf.score(X_train, y_train)))
print('Test set score for Random Forest model: {:.4f}'.format(rf.score(X_test, y_test)))


# ### The training-set accuracy score and the test-set accuracy score are very close to each other in both the models. These two values are quite comparable. So, there is no question of overfitting.

# ## Classification Reports for both the models

# In[41]:


#For logistic Regression Classifier
from sklearn.metrics import classification_report
print(classification_report(y_test, lr.predict(X_test)))


# In[43]:


#For Random Forest Classifier
print(classification_report(y_test, rf.predict(X_test)))


# ## Adjusting the threshold level 
# - store the the first 10 predicted probabilities of two classes- 0 and 1 probabilities in dataframe
# * predict_proba method gives the probabilities for the target variable(0 and 1) in this case, in array form

# In[44]:


#with logistic Regression model
pd.DataFrame(data=lr.predict_proba(X_test)[0:10], columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])


# In[45]:


#with random forest model
pd.DataFrame(data=rf.predict_proba(X_test)[0:10], columns=['Prob of - No rain tomorrow (0)', 'Prob of - Rain tomorrow (1)'])


# ## Observations
# - In each row, the numbers sum to 1.
# - There are 2 columns which correspond to 2 classes - 0 and 1.
#     - Class 0 - predicted probability that there is no rain tomorrow.
#     - Class 1 - predicted probability that there is rain tomorrow.
# 
# - Importance of predicted probabilities
#     - We can rank the observations by probability of rain or no rain.
#     
# - predict_proba process
#     - Predicts the probabilities
#     - Choose the class with the highest probability
