# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import sklearn.linear_model as LM
import datetime as dt
from dateutil import relativedelta as rd
from functools import reduce
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as pp

pd.set_option('display.max_columns', None)
og_df = pd.read_csv('weatherAUS.csv')
og_df['Date'] = pd.to_datetime(og_df['Date'], format='%Y-%m-%d')
df = og_df.copy(deep=True)

print(f"Start time of dataset: {df['Date'].min()}")
print(f"End time of dataset: {df['Date'].max()}")
startdate = df['Date'].min()
enddate = df['Date'].max()
delta = rd.relativedelta(enddate, startdate)
print("\nTotal Timespan of Datset:")
print(f'{delta.days} Days, {delta.months} Months, {delta.years} Years')


print("Average MinTemp by Month")
df['month'] = pd.DatetimeIndex(df['Date']).month
temp = df.groupby(by=df['month']).agg({r'MinTemp': ['mean']})
print(temp)
temp.plot(kind='bar')
ax = temp.plot(kind='bar')
fig = ax.get_figure()
fig.savefig('MinTemp.png')


print(f"Number of Unique Cities in the dataset: {len(df.groupby(by=df['Location']))}")


print('Top 5 rainiest cities: ')
print(df.groupby(by=df['Location']).agg({'Rainfall': ['max']}).sort_values(('Rainfall', 'max'), ascending=False).head(5))

print(f"Mean Pressure9am: {df['Pressure9am'].mean()}")
print(f"Mean Pressure3pm: {df['Pressure3pm'].mean()}")
print(f"Mean Humidity9am: {df['Humidity9am'].mean()}")
print(f"Mean Humidity3pm: {df['Humidity3pm'].mean()}")
print(f"Mean Temp9am:     {df['Temp9am'].mean()}")
print(f"Mean Temp3pm:     {df['Temp3pm'].mean()}")
print('\n')
print(f"Min Pressure9am: {df['Pressure9am'].min()}")
print(f"Min Pressure3pm: {df['Pressure3pm'].min()}")
print(f"Min Humidity9am: {df['Humidity9am'].min()}")
print(f"Min Humidity3pm: {df['Humidity3pm'].min()}")
print(f"Min Temp9am:     {df['Temp9am'].min()}")
print(f"Min Temp3pm:     {df['Temp3pm'].min()}")
print('\n')
print(f"Max Pressure9am: {df['Pressure9am'].max()}")
print(f"Max Pressure3pm: {df['Pressure3pm'].max()}")
print(f"Max Humidity9am: {df['Humidity9am'].max()}")
print(f"Max Humidity3pm: {df['Humidity3pm'].max()}")
print(f"Max Temp9am:     {df['Temp9am'].max()}")
print(f"Max Temp3pm:     {df['Temp3pm'].max()}")
print('\n')
print(f"Mode Pressure9am: {df['Pressure9am'].mode()}")
print(f"Mode Pressure3pm: {df['Pressure3pm'].mode()}")
print(f"Mode Humidity9am: {df['Humidity9am'].mode()}")
print(f"Mode Humidity3pm: {df['Humidity3pm'].mode()}")
print(f"Mode Temp9am:     {df['Temp9am'].mode()}")
print(f"Mode Temp3pm:     {df['Temp3pm'].mode()}")
print('\n')
print(f"Standard Deviation Pressure9am: {df['Pressure9am'].std()}")
print(f"Standard Deviation Pressure3pm: {df['Pressure3pm'].std()}")
print(f"Standard Deviation Humidity9am: {df['Humidity9am'].std()}")
print(f"Standard Deviation Humidity3pm: {df['Humidity3pm'].std()}")
print(f"Standard Deviation Temp9am:     {df['Temp9am'].std()}")
print(f"Standard Deviation Temp3pm:     {df['Temp3pm'].std()}")

print("Pearson Correlation:")
print(df[['MinTemp', 'Rainfall']].corr(method='pearson'))
print(df[['MinTemp', 'Rainfall']].corr(method='spearman'))
ax = df[['Rainfall', 'MinTemp']].plot.scatter(x='MinTemp', y='Rainfall')
fig = ax.get_figure()
fig.savefig('MinTemp_Rainfall.png')





# +
cities = ['Bendigo', 'Portland', 'Albany', 'Richmond', 'Katherine']
temp = df.loc[df['Location'].isin(cities)]

Richmond = len(df[df['Location'] == 'Richmond'])
Bendigo = len(df[df['Location'] == 'Bendigo'])
Portland = len(df[df['Location'] == 'Portland'])
Albany = len(df[df['Location'] == 'Albany'])
Katherine = len(df[df['Location'] == 'Katherine'])#Katherine has the fewest number of records

Katherine = df[df['Location'] == 'Katherine']
temp = temp[temp['Date'].between(Katherine['Date'].min(), Katherine['Date'].max())]
temp.sort_values(by='Date')

df2 = temp.groupby("Location")[["Location", "Rainfall"]].head(len(Katherine))
df2.loc[:,"col"] = temp['Date']
df3 = df2.pivot_table(index="col",columns="Location",values="Rainfall")
print('These are the correlations between Cities and rainfall,')
print('a positive correlation indicates that if its raining in one city')
print('then its likely be raining in the correlated city')
print('A negative correlation indicates that if its raining in one city,')
print('then its not likely be raining in the correlated city')
Portland = temp[temp['Location']=='Portland']
print("Pearson Correlation:")
print(df3.corr(method='pearson'))
print("Spearman Correlation")
print(df3.corr(method='spearman'))
ax = df3.plot.scatter(x='Portland', y='Bendigo')
fig = ax.get_figure()
fig.savefig('Cities_and_Rainfall_ScatterPlot.png')
# -

#Correlation for choosing my own features:
print(df[['Temp3pm','Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'Rainfall','RainTomorrow']].corr())

#PreProcess for Logistic Regression:
data = df[['Temp3pm','Humidity3pm', 'Pressure3pm', 'RainTomorrow']].copy(deep=True)
data["RainTomorrow"] = df["RainTomorrow"].map({'Yes':1,'No':0})
data.dropna(inplace=True)
y = data['RainTomorrow']
x = data.drop(['RainTomorrow'], axis=1)
sc = StandardScaler()


#Logistic Regresssion:
TRAIN_X, TEST_X, TRAIN_Y, TEST_Y, = train_test_split(x, y)
classifier = LM.LogisticRegression()
train_x_list = sc.fit_transform(TRAIN_X)
classifier.fit(train_x_list,TRAIN_Y)
test_x_list = sc.fit_transform(TEST_X)
pred = classifier.predict(test_x_list)

print("The features that I used were: Temp3pm, Humidity3pm, and Pressure3pm")
print(f"My Accuracy was : {(sm.accuracy_score(TEST_Y, pred)*100):2f}%")
print("Confusion Matrix: \n", sm.confusion_matrix(TEST_Y, pred))



#Preprocess
df = og_df.copy(deep=True)
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year
df.drop('Location', axis=1, inplace=True)
df.drop('Date', axis=1, inplace=True)
df.drop('WindDir9am', axis=1, inplace=True)
df.drop('WindDir3pm', axis=1, inplace=True)
df.drop('WindGustDir', axis=1, inplace=True)
df["RainTomorrow"] = df["RainTomorrow"].map({'Yes':1,'No':0})
df["RainToday"] = df["RainToday"].map({'Yes':1,'No':0})
df.dropna(inplace=True)
y = df['RainTomorrow']
x = df.drop(['RainTomorrow'], axis=1)
x = x.drop(['RISK_MM'], axis=1)#This is cheating...

#RFE:
classifier = tree.DecisionTreeClassifier()
rfe = RFE(estimator=classifier, n_features_to_select=18)
features = list(x.columns[0: ])
np_features = np.array(features)
rfe_fitted = rfe.fit(x, y)


print(f"Top Features:\n{np_features[rfe_fitted.support_]} ")
x = df[np_features[rfe_fitted.support_]]
sc = StandardScaler()


TRAIN_X, TEST_X, TRAIN_Y, TEST_Y, = train_test_split(x, y)
classifier = LM.LogisticRegression()
train_x_list = sc.fit_transform(TRAIN_X)
classifier.fit(train_x_list,TRAIN_Y)
test_x_list = sc.fit_transform(TEST_X)
pred = classifier.predict(test_x_list)

print("RFE with Logistic Regression: ")
print(f"Features used were: {np_features[rfe_fitted.support_]}")
print(f"My Accuracy was : {(sm.accuracy_score(TEST_Y, pred)*100):2f}%")
print("Confusion Matrix: \n", sm.confusion_matrix(TEST_Y, pred))



#Preprocess
df = og_df.copy(deep=True)
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year
df.drop('Location', axis=1, inplace=True)
df.drop('Date', axis=1, inplace=True)
df.drop('WindDir9am', axis=1, inplace=True)
df.drop('WindDir3pm', axis=1, inplace=True)
df.drop('WindGustDir', axis=1, inplace=True)
df["RainTomorrow"] = df["RainTomorrow"].map({'Yes':1,'No':0})
df["RainToday"] = df["RainToday"].map({'Yes':1,'No':0})
df.dropna(inplace=True)
y = df['RainTomorrow']
x = df.drop(['RainTomorrow'], axis=1)
x = x.drop(['RISK_MM'], axis=1)#This is cheating...

# +
#PCA:
le = pp.LabelEncoder() 
df = og_df.apply(le.fit_transform)  
features = x.columns[0: ]  
np_features = np.array(features) 

rs = np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=rs)

sc = StandardScaler()
scaled_X_train = sc.fit_transform(X_train)
scaled_X_test = sc.transform(X_test)

classifier = tree.DecisionTreeClassifier()

print("PCA with Decision Tree Clasifier:\n")
pca = PCA(n_components=20)
PCA_X_train = pca.fit_transform(scaled_X_train)
PCA_X_test = pca.transform(scaled_X_test)
classifier.fit(PCA_X_train, y_train)
y_pred = classifier.predict(PCA_X_test)
print(f"explained variance per feature: {pca.explained_variance_ratio_}\n")
print(f"num_components = {20}, accuracy = {sm.accuracy_score(y_test, y_pred)}\n")
print(f"Confusion Matrix:\n {sm.confusion_matrix(y_test, y_pred)}")
# -

print("PCA With Logistic Regression")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=rs)
classifier = LM.LogisticRegression()
pca = PCA(n_components=20)
PCA_X_train = pca.fit_transform(scaled_X_train)
PCA_X_test = pca.transform(scaled_X_test)
classifier.fit(PCA_X_train,y_train)
y_pred = classifier.predict(PCA_X_test)
print(f"explained variance per feature: {pca.explained_variance_ratio_}\n")
print(f"num_components = {20}, accuracy = {sm.accuracy_score(y_test, y_pred)}\n")
print(f"Confusion Matrix:\n {sm.confusion_matrix(y_test, y_pred)}")

print("The best that I was able to get was ~85% using RFE and Logistic Regression")


