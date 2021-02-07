# example code from

from flask import Flask, jsonify, request, make_response
import pandas as pd
import numpy as np
import sklearn
import pickle
import json
from configs import *
from helper_functions import preprocess
from flask import Flask, jsonify, request, make_response
import pandas as pd
import numpy as np
import sklearn
import pickle
import json
from configs import *
from helper import preprocess

app= Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    return jsonify("hello!")

@app.route("/predictions", methods=["GET"])
def predictions():
    data = request.get_json()
    data = pd.DataFrame(data['data'])
    try:
        preprocessed_X=preprocess(data)
        print(preprocessed_X)
        print(preprocessed_X.shape)
    except:
        return jsonify("Error occured while preprocessing your data")
    filename='model_xgb.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    try:
        predictions= loaded_model.predict(preprocessed_X)
        predictions = predictions.tolist()
    except:
        return jsonify("Error occured while processing your data into our model!")
    print("done")
    response={'data':[],'prediction_label':{'promotion':1,'not promotion':0}}
    response['data']=list(predictions)
    return make_response(jsonify(response),200)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)