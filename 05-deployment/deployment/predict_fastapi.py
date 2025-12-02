import pickle
from fastapi import FastAPI, Request



# Load the model
model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in) #Unpacking the pipeline

# Function to make prediction given data for a single customer
def predict_single(customer, dv, model):
  X = dv.transform([customer]) 
  y_pred = model.predict_proba(X)[0, 1]
  return y_pred

app = FastAPI(title="lead-score-prediction")

@app.post("/predict")
async def predict(request: Request):
    customer = await request.json()
    prediction = predict_single(customer, dv, model)
    return {"lead_score": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9696)