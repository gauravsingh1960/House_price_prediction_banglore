import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import util
from util import get_location_names,get_estimated_price

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/get_location')
def get_location():
    response = jsonify({

        'locations': util.get_location_names()
    })
    response.headers.add("access-control-allow-origin", '*')
    return response

@app.route('/predict',methods=['POST'])
def predict():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location,total_sqft,bhk,bath)
    })

    response.headers.add("access control allow origin", '*')
    return response

if __name__ == "__main__":
    print("starting python flask server..")
    util.load_saved_artifacts()
    app.run(debug=True)