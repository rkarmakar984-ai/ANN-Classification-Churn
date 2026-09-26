# 📊 Customer Churn Prediction using Deep Learning

This project is my **first Deep Learning model**, built to predict whether a bank customer is likely to churn based on their demographic, financial, and account-related information.

The project uses a **feed-forward Artificial Neural Network (ANN)** built with **TensorFlow/Keras** and deployed through an interactive **Streamlit web application**.

---

## 📌 Project Overview

Customer churn prediction is a classification problem where the objective is to identify customers who are likely to leave a bank.

The model takes customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Account Balance
* Number of Products
* Credit Card Status
* Active Membership Status
* Estimated Salary

and produces a **probability of customer churn**.

For example:

```text
Churn Probability: 78.34%
```

The application then uses a probability threshold of **0.50** to classify the customer:

```text
Probability >= 0.50 → Likely to Churn
Probability <  0.50 → Not Likely to Churn
```

---

## 🧠 Model Architecture

The prediction model is a **feed-forward Artificial Neural Network (ANN)** implemented using TensorFlow/Keras.

The overall workflow is:

```text
Customer Data
      ↓
Data Preprocessing
      ↓
Label Encoding
      ↓
One-Hot Encoding
      ↓
Feature Scaling
      ↓
Artificial Neural Network
      ↓
Churn Probability
      ↓
Classification
```

### Preprocessing

The following preprocessing techniques were used:

* **Label Encoding** for Gender
* **One-Hot Encoding** for Geography
* **Feature Scaling** using a fitted scaler

The same preprocessing objects used during training are saved and loaded during inference to ensure consistency between training and prediction.

---

## 📈 Model Performance

The neural network was trained for up to **100 epochs**, with validation data used to monitor performance on unseen samples during training.

### Best Validation Results

| Metric                 |     Result | Epoch |
| ---------------------- | ---------: | ----: |
| Validation Accuracy    | **86.75%** |    16 |
| Validation AUC         | **0.8605** |    24 |
| Lowest Validation Loss | **0.3339** |    20 |

### Performance at Epoch 30

| Metric   | Training | Validation |
| -------- | -------: | ---------: |
| Accuracy |   85.71% |     86.40% |
| AUC      |   0.8563 |     0.8595 |
| Loss     |   0.3495 |     0.3350 |

### Training Observations

The model showed significant improvement during the early training epochs.

* Validation accuracy increased from **83.95% in Epoch 1** to a maximum of **86.75% in Epoch 16**.
* Validation AUC increased from **0.8134 in Epoch 1** to **0.8605 in Epoch 24**.
* Validation loss decreased from **0.3985 in Epoch 1** to a minimum of **0.3339 in Epoch 20**.
* After approximately Epoch 15–20, the validation metrics became relatively stable.
* Training and validation performance remained relatively close throughout the first 30 epochs.

> **Note:** These are validation metrics obtained during training and should not be considered final test-set performance.

---

## 📊 Evaluation Metrics

For a complete evaluation of the final model, the following metrics can be calculated on a separate test dataset:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

Accuracy alone may not provide a complete picture for a churn prediction problem. Precision and recall can provide additional insight into how effectively the model identifies customers who are likely to churn.

---

## 🖥️ Streamlit Application

The trained model is integrated into a Streamlit application that allows users to enter customer information and receive a churn prediction.

The application:

1. Collects customer information.
2. Encodes categorical variables.
3. Applies the trained scaler.
4. Passes the processed data to the neural network.
5. Generates a churn probability.
6. Displays the prediction and classification.

Example:

```text
Churn Probability: 78.34%

⚠️ The customer is likely to churn.
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **TensorFlow**
* **Keras**
* **Scikit-learn**
* **Streamlit**
* **Pickle**

---

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── model.h5
├── scaler.pkl
├── onehot_encoder_geo.pkl
├── label_encoder_gender.pkl
├── requirements.txt
└── README.md
```

### File Description

| File                       | Description                        |
| -------------------------- | ---------------------------------- |
| `app.py`                   | Streamlit application              |
| `model.h5`                 | Trained TensorFlow/Keras ANN model |
| `scaler.pkl`               | Saved feature scaler               |
| `onehot_encoder_geo.pkl`   | Saved Geography one-hot encoder    |
| `label_encoder_gender.pkl` | Saved Gender label encoder         |
| `requirements.txt`         | Python dependencies                |
| `README.md`                | Project documentation              |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd Customer-Churn-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv ann_venv
```

### 4. Activate the virtual environment

### Windows

```bash
ann_venv\Scripts\activate
```

### Linux / macOS

```bash
source ann_venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🎯 What I Learned

This project was an important milestone in my transition from **Machine Learning to Deep Learning**.

Through this project, I learned about:

* Building an Artificial Neural Network using TensorFlow/Keras
* Preparing tabular data for Deep Learning
* Encoding categorical variables
* Feature scaling
* Training a neural network
* Monitoring training and validation metrics
* Understanding accuracy, loss, and ROC-AUC
* Saving and loading trained models
* Saving preprocessing objects
* Performing inference using a trained model
* Building an interactive Streamlit application
* Creating an end-to-end Deep Learning deployment pipeline

---

## 🔮 Future Improvements

Possible improvements for this project include:

* Hyperparameter tuning
* Experimenting with different ANN architectures
* Implementing Early Stopping
* Adding a confusion matrix
* Adding ROC and Precision-Recall curves
* Evaluating the model on a separate test dataset
* Optimizing the classification threshold
* Handling class imbalance
* Improving the Streamlit UI
* Adding model explainability using techniques such as SHAP
* Deploying the application online

---

## 📚 Project Purpose

This project was created as a **learning project** to understand the complete workflow of a Deep Learning application — from data preprocessing and neural network training to model inference and deployment.

It represents my **first step into Deep Learning** and provides a foundation for building more advanced neural-network-based projects in the future.

---

## 👨‍💻 Author

**Rahul Karmakar**

This project is part of my journey toward developing stronger skills in **Machine Learning, Deep Learning, Data Science**.
