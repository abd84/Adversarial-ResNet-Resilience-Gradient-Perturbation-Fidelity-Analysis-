from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


device = torch.device("cpu")  
model = models.resnet18(pretrained=False) 
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  
model.fc = nn.Linear(model.fc.in_features, 10)

# Load the fine-tuned weights from model.pth
model.load_state_dict(torch.load('/Users/abdullah/Desktop/VS/bigdata/assesment/finetuned_resnet18_mnist.pth', map_location=device))  
model = model.to(device)
model.eval()  

# Helper Functions for FGSM and FGSM + Gaussian Noise
def fgsm_attack(model, loss_fn, image, label, epsilon):
    image.requires_grad = True
    output = model(image)
    loss = loss_fn(output, label)
    model.zero_grad()
    loss.backward()
    grad = image.grad.data
    perturbed_image = image + epsilon * grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def fgsm_gaussian_attack(model, loss_fn, image, label, epsilon, sigma=0.1):
    image.requires_grad = True
    output = model(image)
    loss = loss_fn(output, label)
    model.zero_grad()
    loss.backward()
    noise = torch.randn_like(image) * sigma
    perturbed_image = image + epsilon * noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Data Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
])

# Pydantic Model for Request Body
class ImageData(BaseModel):
    image: str  # Base64-encoded image string
    epsilon: float
    sigma: float = 0.1
    label: int  

# Accuracy calculation helper function
def calculate_accuracy(preds, targets):
    correct = (preds == targets).sum().item()
    accuracy = (correct / len(targets)) * 100
    return accuracy

# API Endpoint for Image Attack
@app.post("/predict/")
async def predict_attack(data: ImageData):
    try:
        # Decode the base64-encoded image string
        img_data = base64.b64decode(data.image)
        image = Image.open(io.BytesIO(img_data))
        image = transform(image).unsqueeze(0).to(device)  

        # Get the true class label of the image (from the user input)
        true_label = torch.tensor([data.label]).to(device)

        # Loss function
        loss_fn = nn.CrossEntropyLoss()

        # Make predictions for clean image
        clean_output = model(image)
        clean_pred = clean_output.argmax(dim=1)
        clean_accuracy = calculate_accuracy(clean_pred, true_label)

        print(f"True Label: {data.label}, Predicted: {clean_pred.item()}") 

        # FGSM Attack
        perturbed_image_fgsm = fgsm_attack(model, loss_fn, image, true_label, data.epsilon)
        fgsm_output = model(perturbed_image_fgsm)
        fgsm_pred = fgsm_output.argmax(dim=1)
        fgsm_accuracy = calculate_accuracy(fgsm_pred, true_label)

        print(f"FGSM True Label: {data.label}, FGSM Predicted: {fgsm_pred.item()}")  

        # FGSM + Gaussian Attack
        perturbed_image_gaussian = fgsm_gaussian_attack(model, loss_fn, image, true_label, data.epsilon, data.sigma)
        gaussian_output = model(perturbed_image_gaussian)
        gaussian_pred = gaussian_output.argmax(dim=1)
        gaussian_accuracy = calculate_accuracy(gaussian_pred, true_label)

        print(f"Gaussian True Label: {data.label}, Gaussian Predicted: {gaussian_pred.item()}")  

        # Convert back to PIL Image for response
        perturbed_image_fgsm = transforms.ToPILImage()(perturbed_image_fgsm.squeeze(0).cpu())
        perturbed_image_gaussian = transforms.ToPILImage()(perturbed_image_gaussian.squeeze(0).cpu())
        buf_fgsm = io.BytesIO()
        perturbed_image_fgsm.save(buf_fgsm, format="PNG")
        buf_fgsm.seek(0)

        buf_gaussian = io.BytesIO()
        perturbed_image_gaussian.save(buf_gaussian, format="PNG")
        buf_gaussian.seek(0)

        return {
            "fgsm_attack_image": base64.b64encode(buf_fgsm.getvalue()).decode('utf-8'),
            "gaussian_attack_image": base64.b64encode(buf_gaussian.getvalue()).decode('utf-8'),
            "clean_accuracy": clean_accuracy,
            "fgsm_accuracy": fgsm_accuracy,
            "gaussian_accuracy": gaussian_accuracy
        }

    except Exception as e:
        print(f"Error: {str(e)}")  
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
