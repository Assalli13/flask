
import pandas as pd
import numpy as np
import pickle
import streamlit as st

file1 = open('best_model_B_S.pkl', 'rb')
model = pickle.load(file1)
file1.close()

def predictByClientId(json_):

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

    feature_importances = feature_importances.sort_values('importance',ascending=False)

    # Use the to_json method on the DataFrame to convert it to json
    feature_importances_json = feature_importances.to_json(orient='records')

    return {'prediction': y_pred[0],
            'prediction_proba':y_proba[0][1],
            'feature_importances': feature_importances_json}

def main():
    st.title("API de prédiction de remboursement de crédit")

    SK_ID_CURR = st.text_input("Veuillez saisir le numéro d'identifiant du client :")
    if SK_ID_CURR:
        try:
            SK_ID_CURR = int(SK_ID_CURR)
            json_ = {'SK_ID_CURR': SK_ID_CURR}
            result = predictByClientId(json_)
            st.write(f"La prédiction pour le client {SK_ID_CURR} est {result['prediction']}")
            st.write(f"La probabilité de remboursement du crédit pour le client {SK_ID_CURR} est {result['prediction_proba']}")
            st.write("Importance des caractéristiques :")
            st.write(pd.read_json(result['feature_importances']))
        except:
            st.write("Une erreur s'est produite lors de la prédiction.")
            st.write(traceback.format_exc())

if __name__ == '__main__':
    main()
