import sklearn
import torch
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

#loading digits
img= sklearn.datasets.load_digits()

X=img.images
Y=img.target
# Dividing original with 16 so pixels are from 0-1
X=X/16.0

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(Y, dtype=torch.long)
#training and testing division with stratification
X_train_val,X_test,y_train_val,y_test=train_test_split(X_tensor,y_tensor,test_size=0.15,random_state=42,stratify=y_tensor)
#Validation data division
X_train,X_val,y_train,y_val=train_test_split(X_train_val,y_train_val,test_size=0.15,random_state=42,stratify=y_train_val)

"""
#Training without Augmentation
train_data=TensorDataset(X_train,y_train)
test_data=TensorDataset(X_test,y_test)
val_data=TensorDataset(X_val,y_val)

train_data=DataLoader(train_data,batch_size=32,shuffle=True)

def build_model():

    return nn.Sequential(
        nn.Conv2d(1,8,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128,10),
    )

model= build_model()
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(20):
    model.train()
    total_loss=0.0
    for batch_img,batch_labels in train_data:
        optimizer.zero_grad()
        score=model(batch_img)
        loss=loss_fn(score,batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_labels)
    avg_loss=total_loss/len(train_data.dataset)
torch.save(model.state_dict(), "cnn_digit_model.pth")


model.eval()
test_data=DataLoader(test_data,batch_size=32,shuffle=True)
pred=[]
labels=[]
with torch.inference_mode():
    for batch_img,batch_labels in test_data:
        score=model(batch_img)
        prediction=score.argmax(dim=1)
        pred.extend(prediction.numpy())
        labels.extend(batch_labels.numpy())
acc=accuracy_score(labels,pred)
conf_mt=confusion_matrix(labels,pred)
f1=f1_score(labels,pred,average='macro')
class_count=Counter(y_train.numpy())
majority_count=class_count.most_common(1)[0][1]
baseline_acc=majority_count/len(y_train)

print(f"\n--- CNN Test Results ---")
print(f"Majority-Class Baseline Accuracy: {baseline_acc:}")
print(f"Test Accuracy: {acc}")
print(f"Macro F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_mt}")
"""

print('-'*10)
print('Augmentation')
print('-'*10)
# Using data augmentation for better training
flip_img=torch.flip(X_train,dims=[3])
X_train_aug=torch.cat([X_train,flip_img])

y_train_aug=torch.cat([y_train,y_train])

train_data_aug=TensorDataset(X_train_aug,y_train_aug)
test_dataset_obj=TensorDataset(X_test,y_test)
val_data_aug=TensorDataset(X_val,y_val)

train_data=DataLoader(train_data_aug,batch_size=32,shuffle=True)
test_data=DataLoader(test_dataset_obj,batch_size=32,shuffle=True)
#model
def build_model():

    return nn.Sequential(
        nn.Conv2d(1,8,3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128,10),
    )

model= build_model()
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
# Training
for epoch in range(20):
    model.train()
    total_loss=0.0
    for batch_img,batch_labels in train_data:
        optimizer.zero_grad()
        score=model(batch_img)
        loss=loss_fn(score,batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_labels)
    avg_loss=total_loss/len(train_data.dataset)
torch.save(model.state_dict(), "cnn_digit_model_aug.pth")
test_data=TensorDataset(X_test,y_test)
# Evaluation
model.eval()
test_data_eval_loader=DataLoader(test_dataset_obj,batch_size=32,shuffle=True)
pred=[]
labels=[]
with torch.inference_mode():
    for batch_img,batch_labels in test_data_eval_loader:
        score=model(batch_img)
        prediction=score.argmax(dim=1)
        pred.extend(prediction.numpy())
        labels.extend(batch_labels.numpy())
acc=accuracy_score(labels,pred)
conf_mt=confusion_matrix(labels,pred)
f1=f1_score(labels,pred,average='macro')
class_count=Counter(y_train.numpy())
majority_count=class_count.most_common(1)[0][1]
baseline_acc=majority_count/len(y_train)

print(f"\n--- CNN Test Results ---")
print(f"Majority-Class Baseline Accuracy: {baseline_acc}")
print(f"Test Accuracy: {acc}")
print(f"Macro F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_mt}")

print("-" * 10)
print("Vision Transformer (ViT) Training")
print("-" * 10)


# vit model
def build_vit():
  class SimpleViT(nn.Module):

    def __init__(self):
      super().__init__()
      self.patch_proj = nn.Linear(4, 32)
      self.pos_embed = nn.Parameter(torch.randn(1, 16, 32) * 0.02)
      encoder_layer = nn.TransformerEncoderLayer(
          d_model=32, nhead=2, batch_first=True
      )
      self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
      self.fc = nn.Linear(32, 10)

    def forward(self, x):
      B = x.shape[0]
      # Split 8x8 into 16 unfolded 2 by 2 patch
      x = (
          x.unfold(2, 2, 2)
          .unfold(3, 2, 2)
          .permute(0, 2, 3, 1, 4, 5)
          .reshape(B, 16, 4)
      )
      x = self.patch_proj(x)
      x = x + self.pos_embed
      x = self.transformer(x)
      x = x.mean(dim=1)  # average pooling
      return self.fc(x)

  return SimpleViT()


vit_model = build_vit()
optimizer_vit = torch.optim.Adam(vit_model.parameters(), lr=0.001)

# Training loop
for epoch in range(25):
  vit_model.train()
  total_loss = 0.0
  for batch_img, batch_labels in train_data:
    optimizer_vit.zero_grad()
    score = vit_model(batch_img)
    loss = loss_fn(score, batch_labels)
    loss.backward()
    optimizer_vit.step()
    total_loss += loss.item() * len(batch_labels)
  avg_loss = total_loss / len(train_data.dataset)

# Save the trained model
torch.save(vit_model.state_dict(), "vit_digit_model.pth")

# Evaluating vit model
vit_model.eval()
test_loader_eval = DataLoader(test_dataset_obj, batch_size=32, shuffle=False)
pred_vit = []
labels_vit = []
with torch.inference_mode():
  for batch_img, batch_labels in test_loader_eval:
    score = vit_model(batch_img)
    prediction = score.argmax(dim=1)
    pred_vit.extend(prediction.numpy())
    labels_vit.extend(batch_labels.numpy())

acc_vit = accuracy_score(labels_vit, pred_vit)
conf_mt_vit = confusion_matrix(labels_vit, pred_vit)
f1_vit = f1_score(labels_vit, pred_vit, average="macro")

print(f"\n--- ViT Test Results ---")
print(f"Majority-Class Baseline Accuracy: {baseline_acc}")
print(f"Test Accuracy: {acc_vit}")
print(f"Macro F1 Score: {f1_vit}")
print(f"Confusion Matrix:\n{conf_mt_vit}")
# at first i gave learning rate=0.01  which drastically reduced the accuracy of vit by changing lr to 0.001 a very high accuracy is acheived.