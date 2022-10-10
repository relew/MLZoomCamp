
import pickle

from urllib import response

from flask import Flask
from flask import request
from flask import jsonify

app = Flask("churn_hw")
model_name = "model1.bin"
dictV_name = "dv.bin"

with open(model_name,'rb') as f_in:
    model = pickle.load(f_in)

with open(dictV_name,'rb') as f_in:
    dv = pickle.load(f_in)

@app.route("/predict", methods = ['POST'])
def predict():
    customer = request.get_json()

    customer_dict = dv.transform([customer])
    y_pred = model.predict_proba(customer_dict)[:,1]

    result = { 'hw_customer_churn_probablility': float(y_pred)}
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9999)
