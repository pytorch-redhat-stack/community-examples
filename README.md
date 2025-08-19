

---

# MNIST & Fashion MNIST Classification with PyTorch

This repository contains PyTorch implementations for training, evaluating, and deploying simple feed-forward neural networks on the **MNIST** and **Fashion MNIST** datasets.
It includes:

1. **MNIST** – Handwritten digit classification.
2. **Fashion MNIST (fmnist)** – Clothing article classification (basic training and evaluation).
3. **Fashion MNIST v2 (fmnist2)** – Fashion MNIST classification with extended code and testing.
4. **Streamlit App (appfmnist)** – Interactive web app to upload images and predict Fashion MNIST classes.

---

## 📂 Project Structure

```
.
├── mnist.py        # MNIST training, evaluation, and visualization
├── fmnist.py       # Fashion MNIST training, evaluation, and visualization
├── fmnist2.py      # Fashion MNIST variant with additional test image visualization
├── appfmnist.py    # Streamlit app for Fashion MNIST predictions
├── data/           # MNIST/Fashion MNIST datasets (downloaded automatically)
├── fashion_mnist_model.pth # Saved trained model weights (generated after training)
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install torch torchvision streamlit pillow matplotlib numpy
```

---

## 🖋 How to Run

### 1️⃣ MNIST

Train and evaluate on MNIST:

```bash
python mnist.py
```

* Downloads MNIST dataset automatically.
* Trains a simple feed-forward neural network.
* Prints accuracy on test dataset.
* Displays sample predictions.

---

### 2️⃣ Fashion MNIST (fmnist)

Train and evaluate on Fashion MNIST:

```bash
python fmnist.py
```

* Trains a neural network on Fashion MNIST.
* Saves trained weights to `fashion_mnist_model.pth`.
* Evaluates accuracy.
* Displays predictions on sample test images.

---

### 3️⃣ Fashion MNIST v2 (fmnist2)

Alternate Fashion MNIST pipeline with **sample/random image testing** inside Streamlit:

```bash
python fmnist2.py
```

* Loads trained weights.
* Allows testing on random Fashion MNIST images or uploaded images.

---

### 4️⃣ Fashion MNIST Streamlit App (appfmnist)

Run the web app:

```bash
streamlit run appfmnist.py
```

* Upload your own `.jpg`, `.png`, `.jpeg` images.
* See predicted class in real time.
* Uses the model trained in `fmnist.py`.

---

## 🧠 Model Architecture

All models use the same simple feed-forward architecture:

```
Input Layer:  784 nodes (28x28 pixels)
Hidden Layer1: 128 nodes (ReLU)
Hidden Layer2: 64 nodes (ReLU)
Output Layer:  10 nodes (class scores)
```

---

## 📊 Datasets

* **MNIST** – Handwritten digits (0–9).
* **Fashion MNIST** – Clothing items (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

Both datasets are loaded using:

```python
from torchvision import datasets, transforms
```

and normalized to range `[-1, 1]`.

---

## 🖼 Sample Predictions

Example output from MNIST/Fashion MNIST scripts:

| Image          | Predicted   |
| -------------- | ----------- |
| 🖼 Digit "7"   | 7           |
| 👕 T-shirt/top | T-shirt/top |

---

## 🚀 Future Improvements

* Add convolutional neural network (CNN) for higher accuracy.
* Support batch image predictions in the Streamlit app.
* Deploy app on cloud services (e.g., Streamlit Cloud, Heroku).

---

## 📜 License

This project is licensed under the MIT License.

---


