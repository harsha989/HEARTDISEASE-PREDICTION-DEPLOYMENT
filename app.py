# Importing essential libraries
from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle

# Load the SVM Classifier model
filename = 'clf_svm.pkl'
classifier = pickle.load(open(filename, 'rb'))

app=Flask(__name__)


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
      age=int(request.form['age'])
      sex=int(request.form['sex'])
      cp=int(request.form['cp'])
      restbp=float(request.form['restbp'])
      chol=float(request.form['chol'])
      fbs=float(request.form['fbs'])
      restecg=int(request.form['restecg'])
      thalach=float(request.form['thalach'])
      exang=int(request.form['exang'])
      oldpeak=float(request.form['oldpeak'])
      slope=int(request.form['slope'])
      ca=float(request.form['ca'])
      thal=int(request.form['thal'])

      data = np.array([[age,sex,cp,restbp,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
      my_prediction=classifier.predict(data)

      return render_template('result.html', prediction=my_prediction)




if __name__=='__main__':
    app.run(debug=True)
