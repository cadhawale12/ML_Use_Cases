
""" STEPS 
FastAPI handles requests
Form collects inputs
NumPy formats data
Joblib loads model
Random Forest predicts
Jinja displays UI
"""
# Import FastAPI to build a web app, Form means:✔ “I want to receive input from an HTML form”
from fastapi import FastAPI, Form 
# response will be a web page (HTML), not JSON
from fastapi.responses import HTMLResponse
# Connects Python → HTML page
from fastapi.templating import Jinja2Templates
# Represents browser request
from fastapi import Request
# Used to load saved ML model
import joblib
import numpy as np
# Creates the application., Start Web Server
app = FastAPI()
# Tells FastAPI: ✔ “My HTML files are inside templates folder”
templates = Jinja2Templates(directory="templates")
#Loads trained Random Forest.
model = joblib.load("model.joblib")
# When user opens:http://127.0.0.1:8000/   ✔ This function runs.
@app.get("/", response_class=HTMLResponse)
# Show HTML page
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# When user clicks: ✔ Predict button, ✔ Form is submitted
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,     # Receives input from user
    sepal_length: float = Form(...), 
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
# Convert inputs → ML format.Why double brackets?✔ Model expects 2D array✔ Shape = (1, 4)Meaning:👉 1 sample, 4 features
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    classes = ["Setosa", "Versicolor", "Virginica"]
    result = classes[prediction]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result
    })
