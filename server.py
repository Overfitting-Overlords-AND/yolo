from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image, ImageDraw
import torch
import numpy as np
import io
import constants
from model import Yolov1

app = FastAPI()

# run with - uvicorn server:app --host 0.0.0.0 --port 8080 --reload
@app.on_event("startup")
async def startup_event():
  print("Starting Up")
  model = Yolov1()
  model.load_state_dict(torch.load('./output/epoch_1.pt'))
  model.eval()
  app.state.model = model 

@app.on_event("shutdown")
async def shutdown_event():
  print("Shutting down")

@app.get("/")
def on_root():
  return { "message": "Hello App" }

@app.post("/one_number")
async def one_number(file: UploadFile = File(...)):
  contents = await file.read()
  image_stream = io.BytesIO(contents)
  image = Image.open(image_stream)
  draw = ImageDraw.Draw(image)
  image_grayscale = image.convert("L")
  image_tensor = torch.tensor(np.array(image_grayscale))
  image_mask = (image_tensor > 128).float()
  image_mask = image_mask.unsqueeze(0).unsqueeze(0)  
  bbs = app.state.model(image_mask).reshape(2,7,15)
  for cr in range(2):
    for cc in range(7):
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
