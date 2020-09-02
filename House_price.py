import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df1 = pd.read_csv("datasets_20710_26737_Bengaluru_House_Data.csv")
print(df1)

# print(df1.shape)

print(df1.groupby('area_type')['area_type'].agg('count'))

# dropping the columns not required for calculation
df2 = df1.drop(['area_type','society','balcony','availability'], axis='columns')
print(df2.head())

# finding null values
print(df2.isnull().sum())
# dropping null values
df3= df2.dropna()
print(df3.isnull().sum())

print(df3.shape)
# print unique values of size
print(df3['size'].unique())
# creating a BHK column
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3['bhk'].unique())

# finding error in bhk column
print(df3[df3.bhk > 20])

print(df3['total_sqft'].unique())
# convert total_sqft into a single number

def is_float(x):
	try:
		float(x)
	except:
		return False
	return True

print(df3[~df3['total_sqft'].apply(is_float)].head(10))
# converting float values of total_sqft into int
def convert_sqft_to_num(x):
	tokens = x.split('-')
	if len(tokens) == 2:
		return(float(tokens[0])+float(tokens[1]))/2
	try:
		return float(x)
	except:
		return None

df4= df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
print(df4.head(10))

print(df4.loc[30])

# creating new column price per square feet

df5 = df4.copy()
df5["price_per_sqft"] = df5["price"]*100000/df5["total_sqft"]
print(df5.head())

print(len(df5.location.unique()))

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)

# location with data points with less than 10 variables should be considered as others

print(len(location_stats[location_stats<=10]))
location_stats_less_than_10 = location_stats[location_stats<=10]
print(location_stats_less_than_10)

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))

# removal of outliers containing less area from standard 

df5[df5.total_sqft/df5.bhk<300].head()
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
print(df6.shape)

# based on price per sqrft
print(df6.price_per_sqft.describe())

def remove_pps_outliers(df):
	df_out = pd.DataFrame()
	for key, subdf in df.groupby('location'):
		m=np.mean(subdf.price_per_sqft)
		st=np.std(subdf.price_per_sqft)
		reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
		df_out = pd.concat([df_out,reduced_df],ignore_index=True)
	return df_out

df7 = remove_pps_outliers(df6)
print(df7.shape)

def plot_scatter_chart(df,location):
	bhk2 = df[(df.location==location) & (df.bhk==2)]
	bhk3 = df[(df.location==location) & (df.bhk==3)]
	plt.rcParams['figure.figsize'] = (15,10)
	plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK',s=50)
	plt.scatter(bhk3.total_sqft,bhk3.price, marker='+', color='green', label='3 bhk', s=50)
	plt.xlabel("total Square feet Area")
	plt.ylabel("price")
	plt.title(location)
	plt.legend()

plot_scatter_chart(df7,"Rajaji Nagar")
plt.show()

# now we will remove those 2 BHK aprtments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartmentt
def remove_bhk_outliers(df):
	exclude_indices = np.array([])
	for location, location_df in df.groupby('location'):
		bhk_stats = {}
		for bhk, bhk_df in location_df.groupby('bhk'):
			bhk_stats[bhk] = {
				'mean': np.mean(bhk_df.price_per_sqft),
				'std':np.std(bhk_df.price_per_sqft),
				'count': bhk_df.shape[0]
				}
			for bhk, bhk_df in location_df.groupby('bhk'):
				stats = bhk_stats.get(bhk-1)
				if stats and stats['count']>5:
					exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
	return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
print(df8.shape)

print(df8.bath.unique())
print(df8.bhk.unique())
# removing data points having more bathrooms than bedrooms
df9 = df8[df8.bath<df8.bhk+2]
print(df9.shape)

# dropping unwanted columns

df10 = df9.drop(['size','price_per_sqft'],axis='columns')
print(df10.head(10))

dummies = pd.get_dummies(df10.location)
print(dummies.head(10))

df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
print(df11.head(10))

# since we have created dummy variables we can drop location column
df12 = df11.drop('location', axis='columns')
print(df12)

# creating x & y variables

x=df12.drop('price',axis='columns')
print(x.head())

y = df12.price
print(y.head())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)
lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
print(lr_clf.score(x_test,y_test))

# using k-fold cross validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
print(cross_val_score(LinearRegression(), x,y, cv=cv))

# checking the best regreesion technique for the model

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def find_best_model_using_gridsearchcv(x,y):
	algos = {
		'LinearRegression':{
			'model':LinearRegression(),
			'params':{
				'normalize':[True,False]
			}
		},
		'lasso':{
			'model':Lasso(),
			'params':{
				'alpha':[1,2],
				'selection':['random','cyclic']
			}
		},
		'decision_tree':{
			'model':DecisionTreeRegressor(),
			'params':{
				'criterion':['mse','friedman_mse'],
				'splitter':['best','random']
			}
		}
	}
	scores = []
	cv= ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
	for algo_name, config in algos.items():
		gs= GridSearchCV(config['model'],config['params'], cv=cv, return_train_score=False)
		gs.fit(x,y)
		scores.append({
			'model':algo_name,
			'best_score':gs.best_score_,
			'best_params': gs.best_params_
		})
	return pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(find_best_model_using_gridsearchcv(x,y))

print(x.columns)
print(np.where(x.columns=='2nd Phase Judicial Layout')[0][0])

def predict_price(location,sqft,bath,bhk):
	loc_index = np.where(x.columns==location)[0][0]
	
	X = np.zeros(len(x.columns))
	X[0] = sqft
	X[1] = bath
	X[2] = bhk
	if loc_index >= 0:
		X[loc_index] = 1
	return lr_clf.predict([X])[0]

print(predict_price('1st Phase JP Nagar',1000,2,2))
print(predict_price('1st Phase JP Nagar',1000,3,3))
print(predict_price('Indira Nagar',1000,3,3))

import json
columns = {
	"data_columns":[col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
	f.write(json.dumps(columns))

####  Saving model in system  ####
import pickle
pickle.dump(lr_clf,open('model.pkl','wb'))

####  Load model  ####
lr_clf=pickle.load(open('model.pkl','rb'))

#########################################################################
