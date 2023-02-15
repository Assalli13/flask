#import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import pickle
import gunicorn

app = FastAPI()
#app.mount("/static", StaticFiles(directory="static"), name="static")

file1 = open('best_model_B_S.pkl', 'rb')
model = pickle.load(file1)
file1.close()

class Client(BaseModel):
    SK_ID_CURR: int

@app.post('/predictByClientId', response_class=JSONResponse)
def predictByClientId(request: Request, response: Response, client: Client):
    try:
        data_set = pd.read_csv('X_test_selected')
        data_set['SK_ID_CURR'] = data_set['Unnamed: 0.1']
        client_data = data_set[data_set['SK_ID_CURR'] == client.SK_ID_CURR].drop(['Unnamed: 0'], axis=1)
        y_pred = model.predict(client_data)
        y_proba = model.predict_proba(client_data)

        print("Predicted value: ", y_pred[0])
        print("Predicted probability: ", y_proba[0][0])

        feature_importances = pd.DataFrame(columns=['feature', 'importance'])
        feature_importances['feature'] = client_data.columns
        sum_of_importance = model.feature_importances_.sum()
        feature_importances['importance'] = model.feature_importances_ / sum_of_importance

        # Use the to_json method on the DataFrame to convert it to json
        feature_importances_json = feature_importances.to_json(orient='records')

        return {'prediction': y_pred[0],
                'prediction_proba': y_proba[0][1],
                'feature_importances': feature_importances_json}

    except Exception as e:
        error_msg = traceback.format_exc()
        return {'error': 'Error: ' + str(e) + ' - ' + error_msg}


@app.get('/', response_class=HTMLResponse)
def read_root():
    html_content = """
        <html>
            <head>
                <title>Welcome to the API</title>
            </head>
            <body>
                <h1>Welcome to the API</h1>
                <p>This is the main page of the API.</p>
            </body>
        </html>
    """
    return html_content

if __name__ == '__main__':
    app.run()
