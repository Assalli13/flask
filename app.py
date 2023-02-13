import streamlit as st
from flask import Flask, request, jsonify
import traceback
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# load the saved model
file1 = open('best_model_B_S.pkl', 'rb')
model = pickle.load(file1)
file1.close()

@app.route('/predictByClientId', methods=['POST'])
def predictByClientId():
    if model:
        try:
            json_ = request.json
            sample_size = 10000
            data_set = pd.read_csv('X_test_selected')
            data_set['SK_ID_CURR'] = data_set['Unnamed: 0.1']
            client=data_set[data_set['SK_ID_CURR']==json_['SK_ID_CURR']].drop(['Unnamed: 0'],axis=1)
            y_pred = model.predict(client)
            y_proba = model.predict_proba(client)
            #client['TARGET'] = y_pred
            #print the predicted value and predicted probability

            # Display the results in Streamlit
            st.write("Predicted value: ", y_pred[0])
            st.write("Predicted probability: ", y_proba[0][0])

            #print the predicted value and predicted probability

            #feature_importances = model.feature_importances_
            feature_importances = pd.DataFrame(columns=['feature', 'importance'])
            feature_importances['feature'] = client.columns
            sum_of_importance = model.feature_importances_.sum()
            feature_importances['importance'] = model.feature_importances_/sum_of_importance

            # Display the feature importances in Streamlit
            st.write(feature_importances)

            return jsonify({'prediction': y_pred[0],
                'prediction_proba':y_proba[0][1]})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')

if __name__ == '__main__':
    # Call the predictByClientId function when the button is clicked
    if st.button('Predict'):
        predictByClientId()
