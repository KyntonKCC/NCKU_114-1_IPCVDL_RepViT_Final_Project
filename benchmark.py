import torch
import time
import timm
import model  # Registers the RepViT model from the local 'model' directory

def load_model(model_name, is_repvit=False):
    # 1. Create Model
    # With 'import model', timm can find repvit_m0_9 locally.
    try:
        model = timm.create_model(model_name, pretrained=True)
    except RuntimeError as e:
        if "Unknown model" in str(e):
            print(f"❌ Error: timm cannot find model '{model_name}'.")
            print("Please ensure the 'model' folder exists and contains repvit.py.")
            raise e
        else:
            raise e
            
    model.eval()
    
    # 2. Structural Re-parameterization for RepViT (Branch Fusion)
    if is_repvit:
        try:
            import utils 
            # Call the official fusion function
            utils.replace_batchnorm(model)
        except ImportError:
            print("⚠️ Warning: utils.py not found! Cannot perform fusion.")
        except AttributeError:
             print("⚠️ Warning: replace_batchnorm function not found in utils.py.")
            
    return model

def measure_latency(model, input_tensor, runs=50):
    # 1. Warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 2. Benchmark Loop
    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            model(input_tensor)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    # 3. Calculate Results
    avg_time = (end_time - start_time) / runs
    fps = 1 / avg_time
    return avg_time * 1000, fps