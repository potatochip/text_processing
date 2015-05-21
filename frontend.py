from flask import Flask
import requests

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hey yo!'

# For local development:
app.run(debug=True)

# For public web serving:
# app.run(host='0.0.0.0, port=80')