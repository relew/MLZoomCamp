import pickle

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
import dill

input_file = 'adoption_prediction.bin'

with open(input_file, 'rb') as f_in:
    convert_input,final_model = pickle.load(f_in)

convert_input_undrill = dill.loads(convert_input)

app = Flask('adoption_predict')

@app.route('/adoption_predict', methods = ['POST'])
def predict():
    input_json = request.get_json()

    input_df = convert_input_undrill(input_json)
    input_pred = final_model.predict_proba(input_df)[0,1]
    adoption_pred = input_pred >= 0.5

    result = {
        'adoption_probability': float(input_pred),
        'adoption_prediction': bool(adoption_pred)

    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9999)