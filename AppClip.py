import numpy as np
from PIL import Image
import requests
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import torch



urls = [
"https://c2.staticflickr.com/1/50/133067217_9c27664b97_o.jpg", # river
         "https://farm7.staticflickr.com/4052/4241923013_b7c09bd53b_o.jpg", # man walk in the snow
         "https://farm6.staticflickr.com/2686/4453538282_6e57bb0699_o.jpg", # sunshine with the cloud
         ]
images = [
    Image.open(requests.get(url, stream=True).raw) for url in urls]

model_two = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_two = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

st.title("CLIP Model Image Retrivel")
st.write("Enter a caption and get the closest image")

for idx, img in enumerate(images):
    st.image(img, caption=f"Image {idx+1}")

# Get user input
caption = st.text_input("Enter a caption:")

if caption:
    # Preprocess the input
    inputs = processor_two(text=[caption], images=images, return_tensors="pt", padding=True)
    
    # Get the logits
    outputs = model_two(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)  # probabilities

    # Find the image with the highest probability
    best_image_idx = torch.argmax(logits_per_image).item()
    
    # Display the best matching image
    st.write("Best matching image:")
    st.image(images[best_image_idx], caption=f"Best match for '{caption}'")