from fastai.callback.fp16 import AMPMode
import gradio as gr
from fastai.vision.all import *

learn = load_learner("model.pkl")
labels = learn.dls.vocab

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    max_prob = max(probs)
    
    if max_prob < 0.85:
        return "Model is not confident enough to classify this image."
    
    return {labels[i]: float(probs[i]) for i in range(len(labels)) if probs[i] >= 0.01}

title = "🌍 Intel Image Classifier"
description = (
    "Upload a landscape photo to classify it into one of the six natural scene categories: "
    "**Buildings**, **Forest**, **Glacier**, **Mountain**, **Sea**, or **Street**.\n\n"
    "Try uploading different types of scenes and see how the model reacts!\n\n"
    "Note: If the model is not at least 75% confident, no prediction will be shown.\n"
    "Tip: Use clear images with natural content for best accuracy.\n\n"
)

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs="text",  
    title=title,
    description=description,
    allow_flagging="never",
    examples=[
        ["sample_images/forest.jpg"],
        ["sample_images/building.jpg"],
        ["sample_images/mountain.jpg"]
    ]
).launch()
