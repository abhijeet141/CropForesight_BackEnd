from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cropRecommendation = pd.read_csv('./Crop_recommendation.csv')
better_model = pickle.load(open('./crop_recommendation.pkl', 'rb'))

class CropInfo(BaseModel):
    nitrogen: int
    phosphorus: int
    potassium: int
    temperature: int
    humidity: int
    ph: int
    rainfall: int
    features: list

@app.post('/predict')
async def predict_crop(crop_info: CropInfo):
    nitrogen_value = crop_info.nitrogen
    phosphorus_value = crop_info.phosphorus
    potassium_value = crop_info.potassium
    temperature_value = crop_info.temperature
    humidity_value = crop_info.humidity
    ph_value = crop_info.ph
    rainfall_value = crop_info.rainfall
    features = crop_info.features

    prediction = better_model.predict(pd.DataFrame([[nitrogen_value, phosphorus_value, potassium_value, temperature_value, humidity_value, ph_value, rainfall_value, *features]],
                                                   columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', *cropRecommendation.columns]))
    print(prediction)

    return {"result": prediction[0]}
