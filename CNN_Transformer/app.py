import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
from torch import nn


# MODELs
def build_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128, 10),
    )


class TinyViT(nn.Module):

    def __init__(
            self, embed_dim=32, num_heads=2, num_layers=2, num_classes=10
    ):
        super().__init__()
        self.patch_proj = nn.Linear(4, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = (
            x.unfold(2, 2, 2)
            .unfold(3, 2, 2)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(B, 16, 4)
        )
        x = self.patch_proj(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


@st.cache_resource
def load_models():
    # Load CNN
    cnn = build_cnn()
    try:
        cnn.load_state_dict(
            torch.load("cnn_digit_model.pth", map_location="cpu", weights_only=True)
        )
        cnn.eval()
    except Exception:
        try:
            cnn.load_state_dict(
                torch.load("cnn_digit_model_aug.pth", map_location="cpu", weights_only=True)
            )
            cnn.eval()
        except Exception as e:
            return None, None, f"CNN Load Error: {e}"

    # Load ViT
    vit = TinyViT()
    try:
        vit.load_state_dict(
            torch.load("vit_digit_model.pth", map_location="cpu", weights_only=True)
        )
        vit.eval()
    except Exception as e:
        return cnn, None, f"ViT Load Error: {e}"

    return cnn, vit, None


# Preprocessing
def prepare_digit(photo):
    # STEP 1: CONVERT TO GRAYSCALE & AUTO-CONTRAST
    gray = ImageOps.grayscale(photo)
    gray = ImageOps.autocontrast(gray)

    # STEP 2: CHECK THE BACKGROUND COLOR & INVERT IF NEEDED
    pixels = np.array(gray)
    border = np.concatenate([
        pixels[0],  # Top edge
        pixels[-1],  # Bottom edge
        pixels[:, 0],  # Left edge
        pixels[:, -1]  # Right edge
    ])
    if border.mean() > 127:
        gray = ImageOps.invert(gray)

    # STEP 3: FIND THE DIGIT
    pixels = np.array(gray)
    rows, columns = np.where(pixels > 40)
    if len(rows) < 8:
        raise ValueError("The photo looks blank. Upload one clear digit.")

    # STEP 4: CROP THE DIGIT TIGHTLY
    digit = gray.crop((
        columns.min(),
        rows.min(),
        columns.max() + 1,
        rows.max() + 1
    ))

    # STEP 5: CENTER THE DIGIT WITH PADDING
    side = max(digit.size)
    side = round(side * 1.30)
    centered = Image.new("L", (side, side), color=0)
    left = (side - digit.width) // 2
    top = (side - digit.height) // 2
    centered.paste(digit, (left, top))

    # STEP 6: RESIZE TO 8x8 GRID & NORMALIZE
    prepared_8x8 = centered.resize((8, 8), Image.Resampling.LANCZOS)

    arr_8x8 = np.array(prepared_8x8, dtype=np.float32)
    arr_normalized = arr_8x8 / 255.0
    tensor_img = (
        torch.tensor(arr_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )

    return prepared_8x8, tensor_img



# STREAMLIT

st.title("Handwritten Digit Classifier: CNN vs. ViT")
st.write(
    "Upload a photo of a handwritten digit (range 0-9) to test and compare "
    "both models side-by-side!"
)

cnn_model, vit_model, load_error = load_models()

if load_error or cnn_model is None or vit_model is None:
    st.error(
        f"Could not load model checkpoints. Make sure you have run your training "
        f"script first! Details: {load_error}"
    )
else:
    uploaded_file = st.file_uploader(
        "Choose a PNG or JPEG image...", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            original_photo = Image.open(uploaded_file).convert("RGB")
            prepared_8x8, tensor_input = prepare_digit(original_photo)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploaded Photo")
                st.image(
                    original_photo,
                    caption="Original Upload",
                    width="stretch",
                )

            with col2:
                st.subheader("Prepared 8x8 Input")
                st.image(
                    prepared_8x8,
                    caption="Cropped & Centered 8x8 Grid",
                    width=150,
                    clamp=True,
                )

            # Run inference on both models in eval mode
            with torch.inference_mode():
                # CNN Prediction
                cnn_outputs = cnn_model(tensor_input)
                cnn_probs = torch.softmax(cnn_outputs, dim=1)
                cnn_pred = cnn_outputs.argmax(dim=1).item()
                cnn_conf = cnn_probs[0][cnn_pred].item() * 100

                # ViT Prediction
                vit_outputs = vit_model(tensor_input)
                vit_probs = torch.softmax(vit_outputs, dim=1)
                vit_pred = vit_outputs.argmax(dim=1).item()
                vit_conf = vit_probs[0][vit_pred].item() * 100

            st.markdown("---")
            st.subheader("Model Predictions Comparison")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.success(
                    f"### CNN Prediction\n**Digit: {cnn_pred}**\nConfidence:"
                    f" `{cnn_conf:.1f}%`"
                )

            with res_col2:
                st.info(
                    f"### ViT Prediction\n**Digit: {vit_pred}**\nConfidence:"
                    f" `{vit_conf:.1f}%`"
                )

            # Detailed probability breakdown expander
            with st.expander("See full class probability breakdowns"):
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    st.write("**CNN Probabilities:**")
                    for d, p in enumerate(cnn_probs[0].numpy()):
                        st.write(f"Digit {d}: `{p * 100:.2f}%`")
                with b_col2:
                    st.write("**ViT Probabilities:**")
                    for d, p in enumerate(vit_probs[0].numpy()):
                        st.write(f"Digit {d}: `{p * 100:.2f}%`")

        except ValueError as ve:
            st.warning(f"⚠️ Input Warning: {ve}")
        except Exception as ex:
            st.error(f"❌ An error occurred while processing the file: {ex}")