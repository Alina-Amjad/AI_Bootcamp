# Handwritten Digit Prediction

This handwritten program takes handwritten image from user and  predicts the digit using two models CNN and Vision transformer. The numbers that can be predicted ranges from 0 to 9. It uses the pre-trained data to make prediction. 

## 📊 Model Evaluation Results

### 1. CNN Model (Without Augmentation)
* **Majority-Class Baseline Accuracy:** 10.18%
* **Test Accuracy:** 96.67%
* **Macro F1 Score:** 0.9665

**Confusion Matrix:**
```text
[[27  0  0  0  0  0  0  0  0  0]
 [ 0 27  0  0  0  0  0  0  0  0]
 [ 0  0 27  0  0  0  0  0  0  0]
 [ 0  0  0 28  0  0  0  0  0  0]
 [ 0  2  0  0 24  0  1  0  0  0]
 [ 0  0  0  0  0 27  0  0  0  0]
 [ 0  0  0  0  0  0 27  0  0  0]
 [ 0  0  0  0  0  0  0 27  0  0]
 [ 0  4  0  0  0  0  0  0 22  0]
 [ 0  0  0  0  0  0  0  0  2 25]]
```
### 2. CNN Model (With Data Augmentation)
* **Majority-Class Baseline Accuracy:** 10.18%
* **Test Accuracy:** 95.19%
* **Macro F1 Score:** 0.9516

**Confusion Matrix:**
```text
[[27  0  0  0  0  0  0  0  0  0]
 [ 0 24  0  0  0  0  0  0  0  3]
 [ 0  0 26  0  0  0  1  0  0  0]
 [ 0  0  0 26  0  2  0  0  0  0]
 [ 0  0  0  0 26  0  1  0  0  0]
 [ 0  0  0  0  0 27  0  0  0  0]
 [ 0  0  0  0  0  0 27  0  0  0]
 [ 0  1  0  0  0  1  0 25  0  0]
 [ 0  1  0  1  0  0  0  1 22  1]
 [ 0  0  0  0  0  0  0  0  0 27]]
```
### 3. Vision Transformer (ViT) Model
* **Majority-Class Baseline Accuracy:** 10.18%
* **Test Accuracy:** 91.48%
* **Macro F1 Score:** 0.9159

**Confusion Matrix:**
```text
[[26  0  0  0  0  0  0  0  0  1]
 [ 0 27  0  0  0  0  0  0  0  0]
 [ 0  1 26  0  0  0  0  0  0  0]
 [ 0  0  0 22  0  5  0  0  0  1]
 [ 0  1  0  0 23  0  2  0  0  1]
 [ 1  0  0  0  0 25  0  0  0  1]
 [ 0  0  0  0  0  0 27  0  0  0]
 [ 0  2  0  0  0  2  1 22  0  0]
 [ 0  3  0  0  0  0  0  0 23  0]
 [ 0  1  0  0  0  0  0  0  0 26]]
```

## 📂 Project Structure

```text
├── cnn_vit.py              # Training of both CNN and vision Transformer
├── app.py                  # Streamlit web app
├── cnn_digit_model.pth     # Saved weights of CNN model without augmentation
├── cnn_digit_model_aug.pth # Saved weights of CNN model with augmented data
├── vit_digit_model.pth     # Saved weights of Vision Transformer
└── README.md               # Project documentation
```

## 🛠️ Installation Requirements

Libraries required to run the script.

```bash
pip install torch torchvision scikit-learn streamlit pillow numpy matplotlib
```

## Run Code
For training the model
```bash
python cnn_vit.py
```
For running Streamlit
```bash
streamlit run app.py
```

