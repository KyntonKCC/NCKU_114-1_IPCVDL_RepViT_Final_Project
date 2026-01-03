import streamlit as st
import torch
import timm
import numpy as np
import requests
import json
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import model

# --- Page Setup ---
st.set_page_config(page_title="RepViT Heatmap Analysis", layout="wide")
st.title("AI Vision Analysis: RepViT vs ViT")
st.markdown("### 透過 Grad-CAM 熱力圖，檢視模型是否學到了正確的特徵")

# --- 1. Download ImageNet Labels (To resolve ID-only display) ---
@st.cache_data
def get_imagenet_labels():
    try:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        response = requests.get(url)
        labels = response.json()
        return labels
    except Exception as e:
        st.error(f"Failed to download labels, displaying IDs: {e}")
        return [f"Class {i}" for i in range(1000)]

imagenet_labels = get_imagenet_labels()

# --- 2. Load Model (Fixed distillation weight loading issue) ---
def load_model(model_name, is_repvit=False):
    try:
        if is_repvit:
            # === RepViT: Force manual loading of local weights ===
            model = timm.create_model(model_name, pretrained=False)
            
            # Ensure filename is correct
            checkpoint_path = 'repvit_m0_9_distill_300e.pth'
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                
                model.load_state_dict(checkpoint, strict=False)
                
            except FileNotFoundError:
                st.error(f"❌ Weight file not found: {checkpoint_path}")
                st.warning("Please download the weight file from RepViT GitHub and place it in this folder.")
                return None
            except Exception as e:
                st.error(f"❌ RepViT weight loading failed: {e}")
                return None

        else:
            # === ViT: Auto-download via timm ===
            model = timm.create_model(model_name, pretrained=True)
            
        model.eval()
        return model

    except Exception as e:
        st.error(f"Model creation failed: {e}")
        return None

# --- 3. Image Preprocessing (Key Fix: Added Normalization) ---
def preprocess_image(image):
    # ImageNet normalization parameters (Crucial for high accuracy)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) 
    ])
    return preprocess(image).unsqueeze(0)

# --- 4. Handle ViT Output (Required for Grad-CAM) ---
def reshape_transform_vit(tensor):
    height = 14
    width = 14
    # Remove class token
    result = tensor[:, 1:, :]
    # Transpose dimensions
    result = result.transpose(1, 2)
    result = result.reshape(tensor.size(0), result.size(1), height, width)
    return result

# --- 5. Generate Heatmap and Prediction ---
def generate_cam_and_pred(model, target_layer, input_tensor, rgb_img, is_vit=False):
    # A. Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        score, idx = torch.topk(probs, 1)
        class_id = idx.item()
        conf = score.item() * 100
        
        # Lookup label
        if class_id < len(imagenet_labels):
            label_name = imagenet_labels[class_id]
        else:
            label_name = f"ID: {class_id}"

    # B. Grad-CAM Heatmap
    cam = GradCAM(
        model=model, 
        target_layers=target_layer, 
        reshape_transform=reshape_transform_vit if is_vit else None
    )
    
    # Generate heatmap for this image
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization, f"{label_name} ({conf:.1f}%)"

# --- Interface Design ---
col1, col2 = st.columns([1, 2])

with col1:
    st.info("Please select a test image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    # Fixed comparison between these two models
    model_vit_name = "vit_base_patch16_224"
    model_rep_name = "repvit_m0_9"

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Prepare data:
    # 1. Tensor for model input (Normalized)
    img_tensor = preprocess_image(image)
    
    # 2. Base image for Grad-CAM (0~1 float)
    img_resized = image.resize((224, 224))
    rgb_img = np.float32(img_resized) / 255
    
    with col2:
        st.image(image, caption="Original Image", width=250)

    st.divider()

    if st.button("🔍 Run Heatmap Analysis"):
        c1, c2 = st.columns(2)
        
        # === Left: ViT ===
        with c1:
            st.subheader("Challenger: ViT Base")
            with st.spinner("Processing ViT..."):
                model_vit = load_model(model_vit_name, is_repvit=False)
                if model_vit:
                    # Target layer for ViT
                    target_layer_vit = [model_vit.blocks[-1].norm1]
                    try:
                        res, label = generate_cam_and_pred(model_vit, target_layer_vit, img_tensor, rgb_img, is_vit=True)
                        st.image(res, caption=f"ViT Prediction: {label}", width="stretch")
                    except Exception as e:
                        st.error(f"ViT Failed: {e}")

        # === Right: RepViT ===
        with c2:
            st.subheader("Main Character: RepViT M0.9")
            with st.spinner("Processing RepViT..."):
                model_rep = load_model(model_rep_name, is_repvit=True)
                if model_rep:
                    # Target layer for RepViT (Auto-detect last feature layer)
                    target_layer_rep = None
                    try:
                        target_layer_rep = [model_rep.features[-1]]
                    except:
                        try:
                            target_layer_rep = [list(model_rep.modules())[-1]]
                        except:
                            pass
                    
                    if target_layer_rep:
                        try:
                            res, label = generate_cam_and_pred(model_rep, target_layer_rep, img_tensor, rgb_img, is_vit=False)
                            st.image(res, caption=f"RepViT Prediction: {label}", width="stretch")
                        except Exception as e:
                            st.error(f"RepViT Grad-CAM Failed: {e}")
                    else:
                        st.error("RepViT target layer not found")

        st.success("Analysis Complete! Compare the red regions between the two.")