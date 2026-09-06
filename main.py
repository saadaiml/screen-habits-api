import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

# 1. Create the app
app = FastAPI()

# 2. Let your website (a different address) call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the model ONCE when the server starts.
#    This file has NO feature-name notes attached, so we must
#    write the correct order and names ourselves, below.
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 4. The exact 18 column names, in the exact order the model expects.
#    (Same order the training data used — must not be changed.)
feature_order = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day",
    "weekend_screen_time", "screen_to_sleep_ratio", "social_share_of_screen",
    "gaming_share_of_screen", "weekend_vs_weekday", "notif_per_app_open",
    "leisure_vs_work", "gender", "stress_level", "academic_work_impact"
]

# 5. The "order form" — only the RAW fields a person/website would send.
class UserData(BaseModel):
    age: float
    daily_screen_time_hours: float
    social_media_hours: float
    gaming_hours: float
    work_study_hours: float
    sleep_hours: float
    notifications_per_day: float
    app_opens_per_day: float
    weekend_screen_time: float
    gender: Literal["Female", "Male", "Other"]
    stress_level: Literal["Low", "Medium", "High"]
    academic_work_impact: Literal["Yes", "No"]

# 6. The prediction endpoint
@app.post("/predict")
def predict(data: UserData):

    # Calculate the 6 derived ratio fields from the raw numbers.
    screen_to_sleep_ratio = data.daily_screen_time_hours / data.sleep_hours
    social_share_of_screen = data.social_media_hours / data.daily_screen_time_hours
    gaming_share_of_screen = data.gaming_hours / data.daily_screen_time_hours
    weekend_vs_weekday = data.weekend_screen_time / data.daily_screen_time_hours
    notif_per_app_open = data.notifications_per_day / data.app_opens_per_day
    leisure_vs_work = (data.social_media_hours + data.gaming_hours) / data.work_study_hours

    # Put everything in one dictionary, labeled by name.
    # Category fields stay as plain text — this model handles them internally.
    row = {
        "age": data.age,
        "daily_screen_time_hours": data.daily_screen_time_hours,
        "social_media_hours": data.social_media_hours,
        "gaming_hours": data.gaming_hours,
        "work_study_hours": data.work_study_hours,
        "sleep_hours": data.sleep_hours,
        "notifications_per_day": data.notifications_per_day,
        "app_opens_per_day": data.app_opens_per_day,
        "weekend_screen_time": data.weekend_screen_time,
        "screen_to_sleep_ratio": screen_to_sleep_ratio,
        "social_share_of_screen": social_share_of_screen,
        "gaming_share_of_screen": gaming_share_of_screen,
        "weekend_vs_weekday": weekend_vs_weekday,
        "notif_per_app_open": notif_per_app_open,
        "leisure_vs_work": leisure_vs_work,
        "gender": data.gender,
        "stress_level": data.stress_level,
        "academic_work_impact": data.academic_work_impact,
    }

    # Build a one-row table with columns in the model's exact expected order.
    input_df = pd.DataFrame([row])[feature_order]

    # Ask the model to predict.
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability_of_1": float(probability)
    }
