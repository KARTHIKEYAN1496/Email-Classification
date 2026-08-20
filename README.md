# 📧 Spam Email Detection using BERT

## 🚀 Project Overview

This project implements a **Spam Email Detection System** using **BERT (Bidirectional Encoder Representations from Transformers)**. The model is fine-tuned to classify email or SMS messages into two categories:

* **Spam**
* **Ham (Not Spam)**

The system leverages Natural Language Processing (NLP) and Deep Learning techniques to achieve high classification accuracy and provide reliable spam detection.

---

## 🎯 Objectives

* Detect spam messages automatically.
* Fine-tune a pretrained BERT model for binary text classification.
* Evaluate model performance using multiple metrics.
* Save and deploy the trained model for real-world usage.
* Provide an interactive user interface using Gradio.

---

## 🛠️ Technologies Used

| Technology                | Purpose                        |
| ------------------------- | ------------------------------ |
| Python                    | Programming Language           |
| Pandas                    | Data Processing                |
| NumPy                     | Numerical Operations           |
| PyTorch                   | Deep Learning Framework        |
| Hugging Face Transformers | BERT Model                     |
| Scikit-learn              | Data Splitting & Evaluation    |
| Matplotlib                | Visualization                  |
| Seaborn                   | Confusion Matrix Visualization |
| Gradio                    | Web Interface                  |

---

## 📂 Dataset

The dataset contains labeled messages with two classes:

| Label | Meaning                                    |
| ----- | ------------------------------------------ |
| Spam  | Unwanted promotional or fraudulent message |
| Ham   | Legitimate message                         |

### Sample Data

| Label | Message                                 |
| ----- | --------------------------------------- |
| Spam  | Congratulations! You won a free iPhone. |
| Ham   | Meeting is scheduled at 10 AM tomorrow. |

---

## 🔄 Project Workflow

### 1. Data Loading

* Load dataset using Pandas.
* Explore data structure and class distribution.

### 2. Data Preprocessing

* Remove unnecessary columns.
* Handle missing values.
* Encode labels:

  * Ham → 0
  * Spam → 1

### 3. Train-Test Split

* Split dataset into:

  * Training Set (80%)
  * Validation Set (20%)

### 4. Tokenization

* Use `BertTokenizer` from Hugging Face.
* Convert text into:

  * Input IDs
  * Attention Masks

### 5. Dataset Creation

* Create custom PyTorch Dataset class.
* Use DataLoader for batch processing.

### 6. Model Training

* Load pretrained:

  * `bert-base-uncased`
* Fine-tune for binary classification.
* Optimizer:

  * AdamW
* Learning Rate:

  * 2e-5
* Epochs:

  * 3

### 7. Evaluation

Evaluate model using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### 8. Model Saving

Save trained model and tokenizer using:

```python
model.save_pretrained("./spam_bert_model")
tokenizer.save_pretrained("./spam_bert_model")
```

### 9. Deployment

Create a Gradio interface for real-time spam prediction.

---

## 🧠 Model Architecture

Input Email Text

↓

BERT Tokenizer

↓

BERT Base Uncased

↓

Dropout Layer

↓

Classification Head

↓

Spam / Ham Prediction

---

## 📊 Results

### Training Loss

| Epoch | Loss   |
| ----- | ------ |
| 1     | 0.0737 |
| 2     | 0.0173 |
| 3     | 0.0122 |

The model achieved excellent performance with a significant reduction in training loss across epochs.

---

## 📈 Evaluation Metrics

* Accuracy Score
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

These metrics help measure the effectiveness of spam detection and identify classification errors.

---

## 🔍 Sample Predictions

### Example 1

**Input:**

```
Congratulations! You have won a $1000 gift card. Click here now.
```

**Prediction:**

```
Spam
```

---

### Example 2

**Input:**

```
Hi John, the meeting has been moved to 3 PM.
```

**Prediction:**

```
Ham
```

---

## ▶️ Running the Project

### Install Dependencies

```bash
pip install transformers torch pandas numpy scikit-learn matplotlib seaborn gradio
```

### Run Training

```bash
python train.py
```

### Launch Gradio App

```bash
python app.py
```

---




## 🏁 Conclusion and Summary

In this project, we successfully built and fine-tuned a **BERT-based Spam Detection** model.

### Key Achievements:
- **High Performance:** The model reached an accuracy of over **99%** on the validation set.
- **Robustness:** With an F1-score of **0.9695**, the model shows a strong balance between precision and recall, effectively minimizing both false positives and false negatives.
- **Deployment Ready:** We implemented a real-time prediction function and a **Gradio web interface** for easy user interaction.

### Insights from Error Analysis:
By inspecting the 9 incorrect predictions, we noticed that errors often occur in ambiguous cases (e.g., short texts with phone numbers or jokes). Future improvements could involve training on a larger, more diverse dataset or increasing the sequence length during tokenization.

