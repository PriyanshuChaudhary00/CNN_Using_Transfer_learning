import torch
from model import get_model
from PIL import Image
from torchvision.models import VGG16_Weights

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = get_model()
model.load_state_dict(torch.load("checkpoint/cat_vs_dog_model.pth"))
model = model.to(device)
model.eval()

def prediction(path):
    weights = VGG16_Weights.DEFAULT
    transform = weights.transforms()
    image = Image.open(path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(image)
        probability = torch.sigmoid(logit).item()
    if probability > 0.5:
        label = "Dog"
        confidence = probability
    else:
        label = "Cat"
        confidence = 1 - probability        
    return label , confidence

label , confidence = prediction("images/cat2.jpg")

print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2%}")

