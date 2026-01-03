import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import streamlit as st
import torch
import pandas as pd
from benchmark import load_model, measure_latency

# --- Page Configuration ---
st.set_page_config(page_title="RepViT Speedometer", layout="wide")

# --- Title and Description ---
st.title("Model Speed Battle: CNN vs. ViT")
st.markdown("### 驗證論文核心：用 CNN 的架構達到 ViT 的性能")

# --- Sidebar: Settings ---
st.sidebar.header("Settings")
model_choice_1 = st.sidebar.selectbox("Challenger", ["vit_base_patch16_224", "swin_tiny_patch4_window7_224", "resnet50"])
model_choice_2 = st.sidebar.selectbox("RepViT", ["repvit_m0_9", "repvit_m1_1", "repvit_m2_3"])
device = st.sidebar.radio("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])

# Prepare Dummy Input
input_tensor = torch.randn(1, 3, 224, 224).to(device)

if st.button("🔥 Start Benchmark"):
    
    col1, col2 = st.columns(2)
    
    # --- 1. Test Baseline Model ---
    with col1:
        st.subheader(f"Challenger: {model_choice_1}")
        with st.spinner(f"Loading and testing {model_choice_1}..."):
            model1 = load_model(model_choice_1, is_repvit=False).to(device)
            latency1, fps1 = measure_latency(model1, input_tensor)
        st.success("Done!")
        st.metric(label="Latency (Lower is better)", value=f"{latency1:.2f} ms")
        st.metric(label="FPS (Higher is better)", value=f"{fps1:.2f}")

    # --- 2. Test RepViT Model ---
    with col2:
        st.subheader(f"RepViT: {model_choice_2}")
        with st.spinner(f"Loading and testing {model_choice_2}..."):
            model2 = load_model(model_choice_2, is_repvit=True).to(device) 
            latency2, fps2 = measure_latency(model2, input_tensor)
        st.success("Done!")
        st.metric(label="Latency", value=f"{latency2:.2f} ms", delta=f"{latency1-latency2:.2f} ms (Faster)")
        st.metric(label="FPS", value=f"{fps2:.2f}", delta=f"{fps2-fps1:.2f} (Boost)")

    # --- 3. Visualization ---
    st.markdown("---")
    st.subheader("📊 Performance Visualization")
    
    chart_data = pd.DataFrame({
        'Model': [model_choice_1, model_choice_2],
        'FPS': [fps1, fps2],
        'Latency (ms)': [latency1, latency2]
    })
    
    # Draw bar chart
    st.bar_chart(chart_data, x='Model', y='FPS', color='Model')

    # --- Conclusion ---
    speedup = fps2 / fps1
    st.info(f"Conclusion: **{model_choice_2}** is **{speedup:.1f}x** faster than **{model_choice_1}**!")