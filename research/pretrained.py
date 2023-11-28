import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import requests

# Function to load an image from a URL
def load_image(url):
    response = requests.get(url)
    img = Image.open(response.raw)
    return img

# Load a pre-trained YOLO model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model = torch.hub.load('./yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.eval()

# Define image transformations
transform = T.Compose([T.Resize(640), T.ToTensor()])

# Load an image
image_url = "https://example.com/your-image.jpg"  # Replace with your image URL
img = load_image(image_url)
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Perform object detection
with torch.no_grad():
    predictions = model(img_tensor)

# Draw bounding boxes and labels on the image
draw = ImageDraw.Draw(img)
for prediction in predictions.pred[0]:
    x1, y1, x2, y2, conf, cls = prediction
    if conf > 0.5:  # Confidence threshold
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
        draw.text((x1, y1), f'{model.names[int(cls)]}: {conf:.2f}', fill="red")

# Display the image
img.show()
