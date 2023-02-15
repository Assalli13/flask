import streamlit as st
import traceback
import pandas as pd
import numpy as np
import pickle

file1 = open('best_model_B_S.pkl', 'rb')
model = pickle.load(file1)
file1.close()
import streamlit as st

@st.cache
def load_data():
    sample_size = 10000
    data_set = pd.read_csv('X_test_selected')
    data_set['SK_ID_CURR'] = data_set['Unnamed: 0.1']
    return data_set

def predictByClientId():

    st.header("Prediction by Client ID")
    
    json_ = st.json(request.data)
    data_set = load_data()
    client=data_set[data_set['SK_ID_CURR']==json_['SK_ID_CURR']].drop(['Unnamed: 0'],axis=1)
    y_pred = model.predict(client)
    y_proba = model.predict_proba(client)
    
    st.write("Predicted value: ", y_pred[0])
    st.write("Predicted probability: ", y_proba[0][0])

    feature_importances = pd.DataFrame(columns=['feature', 'importance'])
    feature_importances['feature'] = client.columns
    sum_of_importance = model.feature_importances_.sum()
    feature_importances['importance'] = model.feature_importances_/sum_of_importance

    # Use the to_json method on the DataFrame to convert it to json
    feature_importances_json = feature_importances.to_json(orient='records')
    
    st.write("Feature importances:")
    st.write(feature_importances)
    
    return {'prediction': y_pred[0], 'prediction_proba': y_proba[0][1], 'feature_importances': feature_importances_json}

def main():
    st.set_page_config(page_title="Welcome to the API")
    st.write("Welcome to the API")
    
    result = predictByClientId()
    st.write(result)

if __name__ == '__main__':
    main()
