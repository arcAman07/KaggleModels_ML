import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
name = ''
class Water(BaseModel):
    ph:float
    Hardness:float
    Solids:float
    Chloramines:float
    Sulfate:float
    Conductivity:float

app = FastAPI()

with open("C:/Users/amans/clf.pkl", "rb") as f:

    model = pickle.load(f)
@app.get('/')
def index():
     return {'message': 'This is the homepage of the API '}
@app.post('/prediction')
def get_prediction(data:Water):
    received = data.dict()
    ph = received['ph']
    Hardness = received['Hardness']
    Solids = received['Solids']
    Chloramines = received['Chloramines']
    Sulfate = received['Sulfate']
    Conductivity = received['Conductivity']
    pred_name = model.predict([[ph,Hardness,Solids,Chloramines,Sulfate,Conductivity]]).tolist()
    if pred_name[0] == 0:
        name = 'Water is not fit for drinking'
    else:
        name = 'Water is fit for drinking'
    return {'prediction': pred_name}



if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000,debug=True)
