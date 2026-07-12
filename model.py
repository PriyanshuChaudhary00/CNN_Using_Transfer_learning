import torch.nn as nn
import torchvision.models as models

def get_model():
    
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    
    for params in model.features.parameters():
        params.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Linear(25088 , 1024),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(1024 , 512),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(512 , 32),
        nn.ReLU(),
        nn.Linear(32 , 1)
    )

    return model
