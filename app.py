from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to your first machine learning project"
