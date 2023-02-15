from os import system
from flask import Flask, request, jsonify, send_from_directory
import traceback
import pandas as pd
import numpy as np
import pickle

file1 = open('best_model_B_S.pkl', 'rb')
model = pickle.load(file1)
file1.close()
app = Flask(__name__)

@app.route('/predictByClientId', methods=['POST'])
def predictByClientId():

    
    json_ = request.json
    sample_size = 10000
    data_set = pd.read_csv('X_test_selected')
    data_set['SK_ID_CURR'] = data_set['Unnamed: 0.1']
    client=data_set[data_set['SK_ID_CURR']==json_['SK_ID_CURR']].drop(['Unnamed: 0'],axis=1)
    y_pred = model.predict(client)
    y_proba = model.predict_proba(client)
    
            
    print("Predicted value: ", y_pred[0])
    print("Predicted probability: ", y_proba[0][0])

            


    feature_importances = pd.DataFrame(columns=['feature', 'importance'])
    feature_importances['feature'] = client.columns
    sum_of_importance = model.feature_importances_.sum()
    feature_importances['importance'] = model.feature_importances_/sum_of_importance

    # Use the to_json method on the DataFrame to convert it to json
    feature_importances_json = feature_importances.to_json(orient='records')
            
    return jsonify({'prediction': y_pred[0],
            'prediction_proba':y_proba[0][1],
            'feature_importances': feature_importances_json})

@app.route('/')
def index():
    return "Welcome to the API"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path,
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run()

