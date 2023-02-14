from os import system
from flask import Flask, request, jsonify, send_from_directory
import traceback
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

@app.route('/predictByClientId')
def predictByClientId():
    file1 = open('best_model_B_S.pkl', 'rb')
    model = pickle.load(file1)
    file1.close()

    json_ = request.json
    filename = 'X_test_selected.csv'
    data_set = pd.read_csv(filename)
    data_set['SK_ID_CURR'] = data_set['Unnamed: 0.1']
    #client = data_set[data_set['SK_ID_CURR'] == json_.get('SK_ID_CURR')].drop(['Unnamed: 0'], axis=1) if json_ else None
    #return jsonify({'error': 'Missing or invalid input data'})
    #client = data_set[data_set['SK_ID_CURR'] == json_['SK_ID_CURR']].drop(['Unnamed: 0'], axis=1)
    y_pred = model.predict(data_set.drop(['Unnamed: 0'], axis=1))
    y_proba = model.predict_proba(data_set.drop(['Unnamed: 0'], axis=1))

    feature_importances = pd.DataFrame(columns=['feature', 'importance'])
    feature_importances['feature'] = data_set.drop(['Unnamed: 0'], axis=1).columns
    sum_of_importance = model.feature_importances_.sum()
    feature_importances['importance'] = model.feature_importances_/sum_of_importance
    feature_importances_json = feature_importances.to_json(orient='records')

    results = {'prediction': y_pred[0], 'prediction_proba':y_proba[0][1], 'feature_importances': feature_importances_json}
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3000)
   

    

