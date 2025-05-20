# 2. app.py
import gradio as gr
from fastai.vision.all import *

learn = load_learner("model.pkl")  # our model which we will upload to hugging

labels = learn.dls.vocab

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr.Interface(fn=classify_image,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3),
             title="Intel Image Classifier",
             description="Trained with FastAI on Intel Image Classification Dataset").launch()
