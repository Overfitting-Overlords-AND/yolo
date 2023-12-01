from fastapi import FastAPI, File, UploadFile
import fastapi
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image, ImageDraw
import torch
import numpy as np
import io
import constants
from model import Yolov1
import utilities
import base64

app = FastAPI()

# run with - uvicorn server:app --host 0.0.0.0 --port 8080 --reload
@app.on_event("startup")
async def startup_event():
  print("Starting Up")
  model = Yolov1()

  utilities.load_latest_checkpoint(model)
  model.eval()
  app.state.model = model 

@app.on_event("shutdown")
async def shutdown_event():
  print("Shutting down")

@app.get("/")
def on_root():
  return { "message": "Hello App" }

@app.post("/one_number")
async def one_number(request: fastapi.Request):
  raw = (await request.json())["img"]
  raw = raw.split(',')[1]
  contents = base64.b64decode(raw)
  image_stream = io.BytesIO(contents)
  image = Image.open(image_stream)
  draw = ImageDraw.Draw(image)
  image_grayscale = image.convert("L")
  image_tensor = torch.tensor(np.array(image_grayscale))
  image_mask = (image_tensor > 128).float()
  image_mask = image_mask.unsqueeze(0).unsqueeze(0)  
  bbs = app.state.model(image_mask).reshape(constants.SR,constants.SC,15)
  for cr in range(constants.SR):
    for cc in range(constants.SC):
      c, x, y, w, h = bbs[cr,cc,10:15]
      p = bbs[cr,cc,:10]
      x, y = 100 * (cc + x), 100 * (cr + y)
      w, h = w * 100, h * 100
      if c > constants.CONFIDENCE_THRESHOLD:   
        draw.rectangle((x-w/2,y-h/2,x+w/2,y+h/2), outline='red', fill=None)
        draw.text((x-w/2+2, y-h/2+2), str(torch.argmax(p).item()), fill='red')
  buffer = BytesIO()
  image.save(buffer, format="PNG")
  buffer.seek(0)  # Move to the start of the buffer
  return StreamingResponse(buffer, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


  #   async def one_number(request: fastapi.Request):
  # raw = (await request.json())["img"]
  # raw = raw.split(',')[1]
  # npArr = numpy.frombuffer(base64.b64decode(raw), numpy.uint8)
  # img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)
  # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # grayImage = cv2.resize(grayImage, (28, 28), interpolation=cv2.INTER_LINEAR)
  # npImg = numpy.expand_dims(grayImage, axis=0)
  # npImgTensor = torch.tensor(npImg)
  # npImgTensor = npImgTensor.unsqueeze(dim=0).float()
  # npImgTensor = npImgTensor.view(1, 1, 28, 28)
  # output = app.state.digit(npImgTensor)
  # probabilities = torch.nn.functional.softmax(output, dim=1).detach().numpy().tolist()[0]
  # result = [{"class": str(i), "value": prob} for i, prob in enumerate(probabilities)]
