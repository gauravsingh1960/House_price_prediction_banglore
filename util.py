import json
import pickle
import numpy as np

__location = None
__data_columns = None
__model = None

def get_estimated_price(location,total_sqft,bhk,bath):
	try:
		loc_index = __data_columns.index(location.lower())
	except:
		loc_index = -1

	X = np.zeros(len(__data_columns))
	X[0] = total_sqft
	X[1] = bath
	X[2] = bhk
	if loc_index >= 0:
		X[loc_index] = 1

	return round(__model.predict([X])[0],2)

def get_location_names():
	return __location

def load_saved_artifacts():
	print("loading saved artifacts...start")
	global __data_columns
	global __location

	with open("./columns.json",'r') as f:
		__data_columns = json.load(f)["data_columns"]
		__location = __data_columns[3:]

	global __model
	with open("./model.pkl",'rb') as f:
		__model = pickle.load(f)

	print("loading artifacts done")

if __name__ == '__main__':
	load_saved_artifacts()
	print(get_location_names())
	print(get_estimated_price('1st Phase JP Nagar',1000,2,2))
	print(get_estimated_price('1st Phase JP Nagar',1000,3,3))
	print(get_estimated_price('Indira Nagar',1000,3,3))

