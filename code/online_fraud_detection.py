#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('Online Fraud- Untagged Transactions.csv' , sep = ',', header = 0)
index=df.index 
df.head()


# In[3]:


frd_df=pd.read_csv('Online Fraud- Fraud Transactions.csv' , sep = ',', header = 0)
frd_index=frd_df.index
frd_df.head()


# In[4]:


frd_df.shape


# In[5]:


#convert the type of transaction times to hhmmss 
import time
transaction_times=[]
for i in range (len(df.transactionTime)) :
    transaction_times.append(time.strftime('%H:%M:%S', time.gmtime(df.transactionTime[i])))
tt=pd.DataFrame(transaction_times)
df.transactionTime=tt


# In[6]:


transactionTimes=[]
for i in range (len(frd_df.transactionTime)) :
    transactionTimes.append(time.strftime('%H:%M:%S', time.gmtime(frd_df.transactionTime[i])))
tT=pd.DataFrame(transactionTimes)
frd_df.transactionTime=tT


# In[7]:


#change the type of transaction dates to string 
df.transactionDate=df.transactionDate.astype(str)
frd_df.transactionDate=frd_df.transactionDate.astype(str)


# In[8]:


#sort the transaction data 
df=df.sort_values(by=['accountID','transactionDate','transactionTime'])
df.index=index
df.head()


# In[9]:


df.isnull().sum()


# In[10]:


null_column=['transactionCurrencyConversionRate','transactionMethod','transactionDeviceType','transactionDeviceId',
             'browserType','cardNumberInputMethod','paymentInstrumentNumber','paymentBillingAddress','paymentBillingName',
            'shippingAddress','shippingPostalCode','shippingCity','shippingState','shippingCountry','responseCode',
            'purchaseProductType','accountOwnerName','accountAddress','accountCity','accountOpenDate','sumPurchaseCount30Day']
for i in null_column :
    df.drop(i,axis=1,inplace=True)


# In[11]:


#'True' indicates that the corresponding IP address is associated with a proxy server
#A proxy server. In other words, the user is accessing the system or service through a proxy server instead of connecting
#directly with their true IP address
isProxyIP=df.loc[df.isProxyIP ==True ,'ipState' ].value_counts()
isProxyIP


# In[12]:


import matplotlib.pyplot as plt
plt.barh(isProxyIP.iloc[:30].index, isProxyIP.iloc[:30].values)
plt.xlabel('Frequency')
plt.ylabel('State')
plt.title('Top 30 States where potential fraud risk')
plt.show()


# In[13]:


Code=df.transactionCurrencyCode.value_counts()
Code


# In[14]:


plt.pie(Code.iloc[:4].values, labels = Code.iloc[:4].index, explode = [0.2,0.2,0.2,0.2],colors=['#A52A2A',
                                                                                                '#8FBC8F','#104E8B','#CDCD00'],
       autopct='%.1f%%')
plt.show() 


# In[15]:


import plotly.express as px
Type = df["paymentInstrumentType"].value_counts()
figure = px.pie(df, values=Type.values, names=Type.index, hole =0.4, title="Distribution of Payment Type")
figure.show()


# In[16]:


Type


# In[17]:


plt.pie(df.cardType.value_counts().values, labels =df.cardType.value_counts().index,autopct='%.1f%%')
plt.legend()
plt.show()


# In[18]:


df.cardType.value_counts()


# In[19]:


#Transactions from unregistered users might be considered higher risk, 
df.isUserRegistered.value_counts()


# In[20]:


x=np.array(['False','True'])
plt.bar(x,df.isUserRegistered.value_counts().values, color = "#4CAF50")
plt.show()


# In[21]:


df.localHour.value_counts()   #For example, if a transaction occurred at 3:45 PM in New York (Eastern Time Zone),
#the 'localHour' column for that transaction would have a value of '15', indicating that the transaction happened during
#the 15th hourof the day, which corresponds to 3:00 PM - 4:00 PM.


# In[22]:


import seaborn as sns

sns.distplot(df['localHour'], bins=50)
#There is a maximum distribution of 10 to 20 of Hour


# In[23]:


frd_df.isnull().sum()


# In[24]:


frd_df.drop('transactionDeviceId',axis=1,inplace=True)


# In[25]:


#Tag the data 
cnames = df.columns.tolist()
tran_frd=frd_df[['transactionID']].copy()
tran_frd['Label2'] = 1
tag_df=pd.merge(df,tran_frd,on='transactionID',how='left')
tag_df['Label']=0
tag_df.loc[~tag_df['Label2'].isna(),'Label']=1
if ('Label'not in cnames):
    cnames.append('Label')
tag_df=tag_df[cnames]
tag_df.head()
#0: NonFraud transaction
#1: Fraud transaction 


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt 
# calculate correlation matrix
corr = tag_df.corr()# plot the heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap='coolwarm' ,fmt=".2f")
plt.show()


# In[ ]:





# In[27]:


tag_df.Label.value_counts()


# In[28]:


plt.pie(tag_df.Label.value_counts().values, labels =['NonFraud Transactions','Fraud Transactions'],autopct='%1.1f%%')
plt.title('type of transactions ')
plt.show()


# In[29]:


CrosstabResult=pd.crosstab(index=tag_df.cardType,columns=tag_df.Label)
CrosstabResult


# In[30]:


#Grouped bar chart between CARDTYPE and LABEL  
CrosstabResult.plot.bar()


# In[31]:


#Handling missing values of numerical columns with the specefic implementation of MICE strategy 
numerical_df = tag_df.select_dtypes(include=['number'])
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
mice_imputer = IterativeImputer(max_iter=10, random_state=0)
X_imputed=mice_imputer.fit_transform(numerical_df)
numerical_df= pd.DataFrame(X_imputed, columns=numerical_df.columns)
for i in numerical_df.columns :
    tag_df[i]=numerical_df[i]


# In[32]:


#handling missing values of categorical columns using the mode of the column 
categorical_df = tag_df.select_dtypes(include=['object','bool'])
for i in categorical_df.columns :
    category_mode = categorical_df[i].mode().values[0]
    categorical_df[i].fillna(category_mode, inplace=True)
    tag_df[i]=categorical_df[i]


# In[33]:


tag_df.boxplot(column='transactionAmount',by='Label')


# In[34]:


#Removing outliers: Winsorizing method 
x=numerical_df.drop(columns='Label',axis=1)
#features=X.columns
for i in x.columns:
    upper=tag_df[i].quantile(0.90)
    lower=tag_df[i].quantile(0.10)
    tag_df[i] = np.where(tag_df[i] <lower, lower,tag_df[i])
    tag_df[i] = np.where(tag_df[i] >upper, upper,tag_df[i])
    


# In[35]:


tag_df.boxplot(column='transactionAmount',by='Label')


# In[36]:


#convert the categorical values into numerical values by LabelEncoder 
from sklearn.preprocessing import LabelEncoder
categorical_columns = tag_df.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    tag_df[column] = label_encoder.fit_transform(tag_df[column])


# In[37]:


import matplotlib.pyplot as plt
import scipy.stats as stats
binary_data = tag_df['Label']
continuous_data =tag_df.drop(columns='Label',axis=1)

correlations = {}
for col in continuous_data.columns:
    point_biserial_corr, p_value = stats.pointbiserialr(binary_data, continuous_data[col])
    correlations[col] = point_biserial_corr
correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Point-biserial correlation'])
correlation_df.sort_values(by='Point-biserial correlation',ascending=False)


# In[38]:


# Create a heatmap correlation plot using Seaborn
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, fmt='.3f', linewidths=2)
plt.title("Point-Biserial Correlation")
plt.show()


# In[39]:


corr = continuous_data.corr()# plot the heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap='coolwarm' ,fmt=".2f")
plt.show()


# In[121]:


X=tag_df.drop(columns='Label',axis=1)
Y=tag_df['Label']
original_column_names = X.columns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)


# In[122]:


#splitting the data 
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=42)
print (Y_train.shape , Y_test.shape )


# In[41]:


pip install lightgbm


# In[203]:


#param={'learnin_rate':0.05,'boosting_type':'gbdt','objective':'binary','metric':'auc','num_leaves':93}


# In[211]:


import lightgbm as lgb
model = lgb.LGBMClassifier()
model.fit(X_train, Y_train)


# In[212]:


importance = model.feature_importances_
sorted_indices = np.argsort(importance)[::-1]
# Select the top K features
K = 15
selected_feature_indices = sorted_indices[:K]
selected_features = original_column_names[selected_feature_indices]


# In[213]:


impo_df = pd.DataFrame({'Feature': selected_features, 'Importance': importance[sorted_indices[:K]]})
impo_df = impo_df.sort_values(by='Importance', ascending=False)
impo_df


# In[214]:


selected_features


# In[215]:


X_train_selected = X_train[:, selected_feature_indices]
X_test_selected = X_test[:, selected_feature_indices]


# In[216]:


model.fit(X_train_selected,Y_train)
y_pred = model.predict(X_test_selected)


# In[217]:


data=pd.DataFrame({'Actual':Y_test , 'Prediction':y_pred})
data


# In[222]:


data.Actual.value_counts()


# In[218]:


data.Prediction.value_counts()


# In[219]:


y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(Y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc)


# In[220]:


from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.grid(True)
plt.show()


# In[221]:


# Generate a classification report
print("Classification Report:")
print(classification_report(Y_test, y_pred))


# In[ ]:




