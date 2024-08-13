from flask import Flask
import joblib

app = Flask(__name__)

model = joblib.load('model.joblib')

@app.route("/")
def home():
    return "Welcome to your first machine learning project"


@app.route("/predict")
def predict():
    predictions = model.predict([ [0.22911, 106.5677, 1150.122, 2022.344] ])

    return "genre is %s." % predictions[0]
