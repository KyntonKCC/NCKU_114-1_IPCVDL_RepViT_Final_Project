import streamlit as st
import streamlit.components.v1 as components
import time

# --- Page Setup ---
st.set_page_config(page_title="RepViT Architecture Demo", layout="wide")

# --- 1. Initialize Session State ---
if "mode_radio" not in st.session_state:
    st.session_state["mode_radio"] = "Training Phase"
if "trigger_fuse" not in st.session_state:
    st.session_state["trigger_fuse"] = False

# --- 2. Define Button Callbacks ---
def set_inference_mode():
    st.session_state["trigger_fuse"] = True 
    st.session_state["mode_radio"] = "Inference Phase"

def set_training_mode():
    st.session_state["mode_radio"] = "Training Phase"

# --- 3. Handle Animation Logic ---
if st.session_state["trigger_fuse"]:
    with st.spinner("重參數化 (Merging Kernels)..."):
        time.sleep(1.2) 
    st.session_state["trigger_fuse"] = False 
    st.rerun() 

st.title("RepViT 架構：重參數化 (Structural Re-parameterization)")
st.markdown("### Why is RepViT both 'Accurate' and 'Fast'?")

# --- 4. Define Powerful Charting Function ---
def mermaid_chart(code, height=700): 
    html_code = f"""
    <div class="mermaid" style="display: flex; justify-content: center; width: 100%; height: 100%;">
        {code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
    </script>
    """
    components.html(html_code, height=height, scrolling=True)

# --- Sidebar Controls ---
st.sidebar.header("Control Panel")
mode = st.sidebar.radio(
    "Select Mode", 
    ["Training Phase", "Inference Phase"],
    key="mode_radio"
)

# --- 定義圖表 (Mermaid 語法) ---
mermaid_training = """
graph TD
    %%{init: {'themeVariables': { 'fontSize': '16px'}}}%%
    subgraph Training [訓練階段：多分支結構]
    style Training fill:#f9f9f9,stroke:#333,stroke-width:2px
    
    In[輸入 Input] --> Split{分流}
    
    Split -->|分支 1| C3[3x3 Conv]
    C3 --> BN1[Batch Norm]
    
    Split -->|分支 2| C1[1x1 Conv]
    C1 --> BN2[Batch Norm]
    
    Split -->|分支 3| ID[Identity]
    ID --> BN3[Batch Norm]
    
    BN1 --> Add((⊕ 相加))
    BN2 --> Add
    BN3 --> Add

    Add --> Act[ReLU Activation]
    Act --> Out[輸出 Output]
    
    style C3 fill:#ffcccc,stroke:#333
    style C1 fill:#ccffcc,stroke:#333
    style ID fill:#ccccff,stroke:#333
    style Add fill:#ffffcc,stroke:#333
    style Split fill:#ffffff,stroke:#333
    end
"""

mermaid_inference = """
graph TD
    %%{init: {'themeVariables': { 'fontSize': '16px'}}}%%
    subgraph Inference [推論階段：單路結構]
    style Inference fill:#e6f3ff,stroke:#333,stroke-width:2px
    
    In[輸入 Input] --> Fused[Fused 3x3 Conv]
    Fused --> Act[ReLU Activation]
    Act --> Out[輸出 Output]
    
    style Fused fill:#ff9999,stroke:#333,stroke-width:4px
    style Act fill:#ffffff,stroke:#333
    end
"""

# --- Main Display ---

col1, col2 = st.columns([1, 1.5])

with col1:    
    # === Button Area (Using callbacks) ===
    if mode == "Training Phase":
        st.button("重參數化 (Fuse!)", on_click=set_inference_mode, type="primary")
            
    st.divider()
    
    if mode == "Training Phase":
        st.markdown("### 🐢 During Training")
        st.write("""
        * **多分支結構 (Multi-branch)：**
            * 擁有多條路徑 ($3*3$, $1*1$, Identity) 可以讓梯度更容易傳遞，幫助模型學得更好、更準。
            * **缺點：** 計算量大、記憶體佔用高，速度慢。
        """)
    else:
        st.button("↩️ 重設 (Reset)", on_click=set_training_mode)
            
        st.markdown("### 🐇 During Inference")
        st.write("""
        * **單路結構 (Single-path)：** 
            * 利用數學原理，將所有分支的權重 **「融合」** 成一個單一的 $3*3$ 卷積核。
            * **優點：**
                * **速度極快：** 只有一條路要走。
                * **省記憶體：** 不用存中間產物。
                * **精度不變：** 數學上完全等價！
        """)

with col2:
    if mode == "Training Phase":
        mermaid_chart(mermaid_training, height=750)
    else:
        mermaid_chart(mermaid_inference, height=500)

# --- Mathematical Principle Explanation ---
st.divider()
with st.expander("How it works?"):
    st.latex(r'''
    K_{fused} = K_{3\times3} + \text{pad}(K_{1\times1}) + K_{id}
    ''')
    st.latex(r'''
    b_{fused} = b_{3\times3} + b_{1\times1} + b_{id}
    ''')
    st.write("""
    透過卷積的可加性原理，我們可以在數學上將不同大小的卷積核與 Batch Norm 參數合併。
    """)