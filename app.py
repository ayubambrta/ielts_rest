from flask import Flask, jsonify, request

app = Flask(__name__)
  
@app.route('/', methods = ['GET'])
def home():
    data = "Hello World"
    return jsonify({'data': data})
    
if __name__ == '__main__':
    app.run(debug = True)