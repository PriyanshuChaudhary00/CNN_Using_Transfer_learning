from model import get_model
from dataset import get_Data
import torch.optim as optim
import torch.nn as nn
import torch 

path = "../"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = get_model().to(device)
train_loader , test_loader = get_Data(path=path)

epochs = 10
learning_rate = 0.0001


loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier.parameters() , lr=learning_rate , weight_decay=1e-4)


train_losses = []
for epoch in range(epochs):
  print("start")
  total_loss_in_epoch = 0
  for input , target in train_loader:
    optimizer.zero_grad()
    input , target = input.to(device) , target.unsqueeze(1).float().to(device)

    y_pred = model(input)
    loss = loss_fn(y_pred , target)

    loss.backward()
    optimizer.step()
    total_loss_in_epoch += loss.item()
  total_loss_in_epoch /= len(train_loader)
  train_losses.append(total_loss_in_epoch)
  print(f"Epoch: {epoch+1}, Loss: {total_loss_in_epoch}")
  
torch.save(model.state_dict(), "checkpoint/cat_vs_dog_model.pth")
  
