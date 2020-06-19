from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import pickle

app=Flask(__name__)
@app.route('/')
def home():
    return  render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        try:
            age=float(request.form['age'])
            ed=float(request.form['ed'])
            employ=float(request.form['employ'])
            address=float(request.form['address'])
            income=float(request.form['income'])
            creddebt=float(request.form['creddebt'])
            othdebt=float(request.form['othdebt'])
            pred_args=[age,ed,employ,address,income,creddebt,othdebt]
            pred_args_arr=np.array(pred_args)
            pred_args_arr=pred_args_arr.reshape(1,-1)
            nbc=open('nbc_model.pkl','rb')
            ml_model=joblib.load(nbc)
            model_prediction=ml_model.predict(pred_args_arr)


        except ValueError:
            return 'Please check the entered values'

    return render_template('predict.html',prediction=model_prediction )












if __name__=='__main__':
    app.run()