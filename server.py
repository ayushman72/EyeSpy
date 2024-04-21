from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Global variable to store the data
host="0.0.0.0" #your local host

# Route to accept PUT requests and store the data
@app.route('/video', methods=['PUT'])
def receive_data():
    global data
    data = request.get_json()
    return 'Data received'

# Route to display the stored data
@app.route('/data', methods=['GET'])
def serve_data():
    global data
    return (data)

if __name__ == '__main__':
    app.run(host=host, port=5000)
