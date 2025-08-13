# Iris Classification with Keras

This project implements a simple **Neural Network** in **Keras** to classify flowers from the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) into one of three species:
- Setosa
- Versicolor
- Virginica

Even though we don't manually write the **backpropagation** equations here, Keras (via TensorFlow) automatically performs it during training using **automatic differentiation**.

---

## 📌 Project Overview

- **Framework:** TensorFlow / Keras
- **Task:** Multi-class classification (3 classes)
- **Dataset:** Iris (150 samples, 4 features)
- **Model architecture:**
  - Input layer: 4 neurons (one per feature)
  - Hidden layer: 8 neurons, activation = ReLU
  - Output layer: 3 neurons, activation = Softmax
- **Loss function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metric:** Accuracy

---

## ⚙️ How It Works

1. **Data Loading & Preprocessing**
   - Load the Iris dataset from `sklearn.datasets`.
   - Standardize features with `StandardScaler`.
   - One-hot encode the target labels.

2. **Model Definition**
   ```python
   model = Sequential()
   model.add(Dense(8, input_shape=(4,), activation='relu'))
   model.add(Dense(3, activation='softmax'))

    Compilation

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

Training (Backpropagation)

model.fit(X_train, y_train,
          validation_split=0.2,
          epochs=100,
          batch_size=8)

    Forward pass: The network computes predictions for each batch.

    Loss calculation: Compares predictions to true labels.

    Backpropagation: TensorFlow computes gradients of the loss with respect to every weight and bias using reverse-mode automatic differentiation.

    Weight update: The Adam optimizer applies updates to minimize the loss.

Evaluation

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.3f}")