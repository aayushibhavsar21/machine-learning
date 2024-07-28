from flask import Flask, request, jsonify
from flask_cors import CORS
import util

app = Flask(__name__)
CORS(app)

@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    symptoms = sorted(util.get_symptoms())
    response = jsonify({
        'symptoms' : symptoms
    })
    return response

@app.route('/predict_problem', methods=['POST'])
def predict_problem():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    health_problem = util.predict_problem(*symptoms)
    precautions = util.get_precaution()
    home_remedies = util.get_home_remedies()
    response = jsonify({
        'health_problem': health_problem,
        'precaution': precautions,
        'home_remedies': home_remedies
    })
    return response


if __name__ == '__main__':
    print("Starting Python Flask server for health problem prediction..")
    util.load_saved_data()
    app.run()
