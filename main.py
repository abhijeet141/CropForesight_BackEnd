from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cropRecomendation = pd.read_csv('./Crop_recommendation.csv')
better_model = joblib.load('./crop_recomendation.pkl')

class CropInfo(BaseModel):
    nitrogen: int = Field(..., ge=0)
    phosphorus: int = Field(..., ge=0)
    potassium: int = Field(..., ge=0)
    temperature: int = Field(..., ge=-50, le=50)
    humidity: int = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=0, le=14)
    rainfall: float = Field(..., ge=0)

@app.post('/predict')
async def predict_crop(crop_info: CropInfo):
    try:
        nitrogen_value = crop_info.nitrogen
        phosphorus_value = crop_info.phosphorus
        potassium_value = crop_info.potassium
        temperature_value = crop_info.temperature
        humidity_value = crop_info.humidity
        ph_value = crop_info.ph
        rainfall_value = crop_info.rainfall

        input_data = pd.DataFrame([[nitrogen_value, phosphorus_value, potassium_value,
                                    temperature_value, humidity_value, ph_value, rainfall_value]],
                                    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        print("Input data:", input_data)

        prediction = better_model.predict(input_data)
        crop = cropRecomendation.loc[prediction[0]]

        print("Prediction:", crop["label"])

        crop_details = {
            "crop": crop["crop"],
            "description": crop["description"],
            "soil_type": crop["soil_type"],
            "temperature_range": crop["temperature"],
            "humidity_range": crop["humidity"],
            "ph_range": crop["ph"],
            "rainfall_range": crop["rainfall"]
        }

        print("Crop details:", crop_details)

        return {
            "result": crop["label"],
            "crop_details": crop_details
        }
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Prediction failed. Please check the input data.")
