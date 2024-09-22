from flask import Flask, request
import joblib

app = Flask(__name__)

model = joblib.load('model.joblib')

@app.route("/")
def home():
    return "Welcome to your first machine learning project"


@app.route("/predict")
def guess():
    # request should have a query parameter named data as a csv string, e.g., data=0.2,0.78,12.34
    array_str = request.args.get('data')
    
    if array_str:
        # Split the string by commas to create a list of strings
        data = array_str.split(',')
        
        # Optional: Convert the list of strings to a list of floats
        data = [float(i) for i in data]
    else:
        data = [0.22911, 106.5677, 1150.122, 2022.344]
    
    predictions = model.predict([ data ])

    return "prediction is %s." % predictions[0]
