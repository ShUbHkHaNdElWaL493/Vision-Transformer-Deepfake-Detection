# Vision Transformer for Deepfake Detection (Celeb-DF)

This project implements a **Vision Transformer (ViT)** for deepfake detection in videos, designed to detect deepfakes using the Celeb-DF (v2) dataset.

## 📌 Features

- Vision Transformer (ViT)-based classification
- Frame-wise processing with temporal averaging
- Random frame sampling to reduce memory usage
- Evaluation metrics:
	- Accuracy
	- Area Under Curve (AUC)
	- Precision
	- Equal Error Rate (EER)

## 🛠️ Installation

Clone the repository and install dependencies:
```bash
pip  install  -r  requirements.txt
```

## 📂 Dataset

The model is trained on the **Celeb-DF v2** dataset. You can download it from here: [Celeb-DF GitHub](https://github.com/yuezunli/Celeb-DF)

Just download the dataset and rename the root folder as *dataset*.

## ⚙️ Hardware Recommendation

Try training and testing the model over platforms like [Google Colab](https://colab.research.google.com) and [Kaggle](https://www.kaggle.com) for low time consumption. For training on Kaggle:
- Use **T4 GPU**
- A100 often over-allocates memory, avoid unless needed
- Reduce memory usage by:
	- Lowering frame count per video
	- Using smaller image sizes (e.g., 224x224)
	- Using mixed precision (`torch.cuda.amp`)

## 💡 Tips

- Use `transform = transforms.Compose([...])` for image preprocessing.
- Ensure all tensors are sent to `device` (e.g., `cuda`).
- You can optionally pre-cache video frame tensors if storage allows.
- Save model after training using:
```python
torch.save(model.state_dict(), "model.pth")
```
- And load it later using:
```python
model.load_state_dict(torch.load("model.pth"))
model.eval()
```
- This allows the model to be used for various purposes including real-time operations.

## 📬 Contributing

Feel free to open an issue or submit a pull request if you'd like to improve or extend this project.