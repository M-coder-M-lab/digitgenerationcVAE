import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import CVAE  # モデルの定義を別ファイルにすることを想定

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのロード
latent_dim = 3
num_classes = 10
model = CVAE(latent_dim=latent_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("cvae.pth", map_location=device))
model.eval()

# Streamlit UI
st.title("Conditional Variational Autoencoder (CVAE) Generator")
st.write("### 手書き数字の生成")

# ユーザー入力
digit = st.number_input("生成したい数字 (0～9)", min_value=0, max_value=9, step=1)
n_samples = st.slider("生成する画像の枚数", min_value=1, max_value=10, value=5)

generate_button = st.button("画像を生成")

if generate_button:
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)
        labels = torch.full((n_samples,), digit, dtype=torch.long, device=device)
        generated_images = model.decoder(z, labels)
    
    # 画像の表示
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
    if n_samples == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(generated_images[i].squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')
    
    st.pyplot(fig)
