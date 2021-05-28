import numpy as np
import pandas as pd
import flask
import cython
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_jsonpify import jsonpify
import gzip, pickle, pickletools
filepath = "miniproject.pkl"

app = Flask(__name__)
CORS(app)

model = pickle.load(open('miniproject.pkl','rb'))

@app.route('/heartprediction',methods=['POST'])
def yieldprediction():
    cropdata = request.get_json()
    age = cropdata.get('age')
    sex = cropdata.get('sex')
    cp = cropdata.get('cp')
    trtbps = cropdata.get('trtbps')  
    chol = cropdata.get('chol')

    fbs = cropdata.get('fbs')
    restecg = cropdata.get('restecg')
    thalachh = cropdata.get('thalachh')
    exng = cropdata.get('exng')
    oldpeak = cropdata.get('oldpeak')  
    slp = cropdata.get('slp')
    caa = cropdata.get('caa')  
    thall = cropdata.get('thall')

    cols = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']
    pd_df = pd.DataFrame(columns=cols)
    pd_df.loc[0] = [age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]

    y_pred = model.predict(pd_df)
    return flask.jsonify(output=int(y_pred[0]))

@app.route('/heartprediction1/<age>/<sex>/<cp>/<trtbps>/<chol>/<fbs>/<restecg>/<thalachh>/<exng>/<oldpeak>/<slp>/<caa>/<thall>',methods=['GET'])
def yieldprediction1(age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall):

    cols = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']
    pd_df = pd.DataFrame(columns=cols)
    pd_df.loc[0] = [age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]

    y_pred = model.predict(pd_df)
    return flask.jsonify(output=int(y_pred[0]))

@app.route('/getdistricts/<statename>',methods=['GET'])
def getdistricts(statename):
    return "YES"

if __name__ == "__main__":
    app.run(debug=True)