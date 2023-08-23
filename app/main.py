from fastapi import FastAPI,Request
from app.code import predict_car
import pickle
import requests

app = FastAPI()
url = "http://172.17.0.2:80/api/genhog"
m = pickle.load(open(r'/carBrandClass/model/carModel.pkl','rb'))

@app.get("/")
def root():
    return {"message": "this is car api"}

# http://localhost:80/api/predict_car
@app.get("/api/predict_car")
async def read_img64(request:Request):
    item = await request.json()
    item_img = item["img"] 
    item_img = str.split(str(item_img),",")[1]
    response = requests.get(url,json={"img":item_img})
    hog = response.json()
    brand = predict_car(m,hog['Hog'])
    return {"Brand":brand}