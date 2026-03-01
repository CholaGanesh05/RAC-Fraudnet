import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 1. CONFIGURATION
# Using the raw string (r"") to handle Windows backslashes correctly
CONFIG = {
    "model_name": "distilbert-base-uncased",
    "num_labels": 2,
    "feature_dim": 768,
    "max_len": 128,
    "model_path": r"rac_fraudnet_best.pt" #Adjust the path as needed.
}

# 2. MODEL ARCHITECTURE
class PrototypeDistilBERT(nn.Module):
    def __init__(self, model_name, num_labels, feature_dim):
        super(PrototypeDistilBERT, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.prototypes = nn.Parameter(torch.randn(num_labels, feature_dim))
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        x_exp = cls_embedding.unsqueeze(1)
        p_exp = self.prototypes.unsqueeze(0)
        distances = torch.pow(x_exp - p_exp, 2).sum(dim=2) 
        return -distances

# 3. ASSET LOADING
@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = PrototypeDistilBERT(CONFIG["model_name"], CONFIG["num_labels"], CONFIG["feature_dim"])
    
    if os.path.exists(CONFIG["model_path"]):
        model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
        model.to(device)
        model.eval()
        return model, tokenizer, device
    return None, tokenizer, device

# 4. INTERFACE
st.set_page_config(page_title="RAC-FraudNet Local", page_icon="🛡️")
st.title("🛡️ RAC-FraudNet: Spam Detection")

model, tokenizer, device = load_assets()

if model is None:
    st.error(f"File Not Found: Ensure the model is at {CONFIG['model_path']}")
else:
    user_input = st.text_area("Analyze Message:", placeholder="Paste text here...", height=150)
    
    if st.button("Run Analysis"):
        if user_input.strip():
            with st.spinner("Processing..."):
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding="max_length", max_length=CONFIG["max_len"])
                
                with torch.no_grad():
                    logits = model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_idx = np.argmax(probs)
                
                # Corrected Confidence Logic
                confidence = float(probs[pred_idx]) 
                st.divider()
                
                if pred_idx == 1:
                    st.error(f"**Result: SPAM 🚨**")
                else:
                    st.success(f"**Result: HAM ✅**")
                
                st.write(f"Confidence: **{confidence * 100:.2f}%**")
                st.progress(confidence)