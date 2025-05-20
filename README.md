# 🌍 Intel Image Classifier (FastAI + Gradio)

This project demonstrates an end-to-end image classification pipeline using the **Intel Image Classification Dataset**. A FastAI-based deep learning model is trained to classify images into 6 natural scene categories, and then deployed using **Gradio** for interactive predictions.

## 📁 Dataset

The dataset used is from Kaggle:  
🔗 https://www.kaggle.com/datasets/puneet6060/intel-image-classification

It contains ~25,000 color images across 6 categories:

- 🏙️ Buildings  
- 🌲 Forest  
- 🧊 Glacier  
- 🏔️ Mountain  
- 🌊 Sea  
- 🛣️ Street

Images are typically 150x150 in size and organized into:
/seg_train # Training images
/seg_test # Test images
/seg_pred # Images for inference


---

## 🧠 Model

- Architecture: `ResNet18` (with experiments using ResNet34 and ResNet50)
- Framework: FastAI v2
- Techniques used:
  - Transfer learning (freeze/unfreeze)
  - Discriminative learning rates
  - Manual weight initialization (Xavier)
  - Training visualization and evaluation

**Best Accuracy Achieved**: **94.4%**

---

## 🖥️ Web Demo

The trained model is deployed via [Gradio](https://www.gradio.app/) and hosted on Hugging Face Spaces:  
👉 [Try the Demo](https://huggingface.co/spaces/okanolgun/intel-classifier-model-ada447)

### Features:
- Upload any landscape image for classification
- Displays top 3 predictions with confidence bars
- Hides output if model confidence is below 75%
- Preloaded sample images for quick testing

---

## 🧪 Example Inference Code

```python
learn = load_learner("model.pkl")

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    max_prob = max(probs)
    if max_prob < 0.75:
        return {label: 0.0 for label in learn.dls.vocab}
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}
