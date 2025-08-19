from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.get("/workout-plans")
def get_workout_plans(limit: int = 10, offset: int = 0):
    return {"limit": limit, "offset": offset}


