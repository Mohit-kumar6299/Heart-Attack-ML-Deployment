from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle

app = FastAPI()

class Item(BaseModel):
    Age: float
    Cholesterol: float
    Heart_rate: float = Field(..., alias="Heart rate")
    Diabetes: float
    Family_History: float = Field(..., alias="Family History")
    Smoking: float
    Obesity: float
    Alcohol_Consumption: float = Field(..., alias="Alcohol Consumption")
    Exercise_Hours_Per_Week: float = Field(..., alias="Exercise Hours Per Week")
    Diet: float
    Previous_Heart_Problems: float = Field(..., alias="Previous Heart Problems")
    Medication_Use: float = Field(..., alias="Medication Use")
    Stress_Level: float = Field(..., alias="Stress Level")
    Sedentary_Hours_Per_Day: float = Field(..., alias="Sedentary Hours Per Day")
    Income: float
    BMI: float
    Triglycerides: float
    Physical_Activity_Days_Per_Week: float = Field(..., alias="Physical Activity Days Per Week")
    Sleep_Hours_Per_Day: float = Field(..., alias="Sleep Hours Per Day")
    Blood_sugar: float = Field(..., alias="Blood sugar")
    CK_MB: float = Field(..., alias="CK-MB")
    Troponin: float
    Gender: float
    Systolic_blood_pressure: float = Field(..., alias="Systolic blood pressure")
    Diastolic_blood_pressure: float = Field(..., alias="Diastolic blood pressure")

# Load the trained model
model = pickle.load(open('extra_trees_model.pkl', 'rb'))

@app.post("/predict/")
async def predict(item: Item):
    data = [[
        item.Age,
        item.Cholesterol,
        item.Heart_rate,
        item.Diabetes,
        item.Family_History,
        item.Smoking,
        item.Obesity,
        item.Alcohol_Consumption,
        item.Exercise_Hours_Per_Week,
        item.Diet,
        item.Previous_Heart_Problems,
        item.Medication_Use,
        item.Stress_Level,
        item.Sedentary_Hours_Per_Day,
        item.Income,
        item.BMI,
        item.Triglycerides,
        item.Physical_Activity_Days_Per_Week,
        item.Sleep_Hours_Per_Day,
        item.Blood_sugar,
        item.CK_MB,
        item.Troponin,
        item.Gender,
        item.Systolic_blood_pressure,
        item.Diastolic_blood_pressure
    ]]
    prediction = model.predict(data)

    # Add labels/explanation for the prediction
    if prediction[0] == 0:
        result = "No disease"
    else:
        result = "Disease detected"

    return {
        "prediction": prediction[0],
        "result": result
    }

import uvicorn

if __name__ == "__main__":
    uvicorn.run("jkl:app", host="127.0.0.1", port=8000, reload=True)
