# 📧 SPAM EMAIL DETECTION USING BERT 🚨

🚀 Project Overview

This project implements an AI-powered Spam Email Detection System using BERT (Bidirectional Encoder Representations from Transformers).

The model is fine-tuned to classify text messages into two categories:

- 🚨 Spam — Unwanted, promotional, or fraudulent messages
- ✅ Ham — Legitimate messages

The project combines Natural Language Processing (NLP), Deep Learning, and Transformer-based models to build an effective spam detection system.

A Gradio web interface is also created to allow users to enter a message and receive a real-time prediction with confidence.

---

🎯 Objectives

The main objectives of this project are:

- 📥 Load and preprocess a spam/ham message dataset
- 🧹 Clean and prepare the text data
- 🔄 Convert text labels into numerical values
- ✂️ Split the dataset into training and validation sets
- 🔤 Tokenize messages using the BERT tokenizer
- 🧠 Fine-tune a pretrained BERT model
- 📊 Evaluate model performance
- 📈 Analyze the confusion matrix
- 💾 Save the best-performing trained model
- 🔮 Perform real-time spam predictions
- 🌐 Build an interactive Gradio application

---

📂 Dataset

The project uses a labeled message dataset containing two classes:

Label| Description
🚨 Spam| Unwanted, promotional, or fraudulent message
✅ Ham| Legitimate message

Example Data

Label| Message
🚨 Spam| Congratulations! You have won a free prize.
✅ Ham| Hi, the meeting is scheduled for 10 AM tomorrow.

The dataset is loaded using Pandas and unnecessary columns are removed during preprocessing.

---

🛠️ Technologies Used

Technology| Purpose
🐍 Python| Programming Language
🐼 Pandas| Data Processing
🔢 NumPy| Numerical Operations
🔥 PyTorch| Deep Learning Framework
🤗 Hugging Face Transformers| BERT Model
🤖 Scikit-learn| Data Splitting & Evaluation
📊 Matplotlib| Visualization
📈 Seaborn| Confusion Matrix
🌐 Gradio| Interactive Web Interface
📓 Jupyter Notebook| Development Environment

---

🧠 Model Used

🤗 BERT — "bert-base-uncased"

This project uses the pretrained:

"bert-base-uncased"

model from Hugging Face Transformers.

BERT is a Transformer-based language model capable of understanding the context and relationships between words in a sentence.

The pretrained BERT model is fine-tuned for binary classification.

---

🏗️ Model Architecture

📩 Input Message
       ↓
🔤 BERT Tokenizer
       ↓
🆔 Input IDs + Attention Mask
       ↓
🤗 BERT Base Uncased
       ↓
🔄 Dropout Layer
       ↓
🧠 Classification Head
       ↓
📊 Binary Classification
       ↓
🚨 Spam / ✅ Ham

---

🔄 Project Workflow

1️⃣ Data Loading

The dataset is loaded using Pandas.

df = pd.read_csv("spam.csv", encoding="latin-1")

Unnecessary columns are removed and the remaining columns are renamed:

label
message

---

2️⃣ Data Preprocessing 🧹

The labels are converted into numerical values:

Ham  → 0
Spam → 1

The dataset is also checked for:

- Missing values
- Class distribution
- Dataset shape

---

3️⃣ Train-Validation Split ✂️

The dataset is divided into:

- 📚 80% Training Data
- 🧪 20% Validation Data

Stratified splitting is used to maintain the class distribution.

---

4️⃣ BERT Tokenization 🔤

The "BertTokenizer" is used to convert text into numerical representations.

Configuration:

- 🤗 Model: "bert-base-uncased"
- 📏 Maximum sequence length: "128"
- 🔄 Truncation: Enabled
- 📦 Padding: Enabled

The tokenizer generates:

- 🆔 Input IDs
- 🎭 Attention Masks

---

5️⃣ PyTorch Dataset & DataLoader 🔥

A custom PyTorch "Dataset" class is created for handling tokenized messages.

The DataLoader configuration includes:

Batch Size: 16
Training Shuffle: Yes
Validation Shuffle: No

---

6️⃣ Model Training 🧠

The pretrained BERT model is fine-tuned for binary classification.

Training Configuration

Parameter| Value
🤗 Model| "bert-base-uncased"
🔢 Number of Classes| 2
📚 Epochs| 3
⚙️ Optimizer| AdamW
📉 Learning Rate| "2e-5"
📦 Batch Size| 16
📏 Max Length| 128
🎯 Loss| Cross Entropy

A learning-rate scheduler is also used during training.

---

📉 Training Performance

The training loss decreased significantly across the three epochs:

Epoch| Training Loss
1️⃣| 0.0737
2️⃣| 0.0173
3️⃣| 0.0122

This indicates that the model successfully learned the patterns present in the training dataset.

---

📊 Model Evaluation

The model is evaluated using several performance metrics:

- 🎯 Accuracy
- 🎯 Precision
- 🔄 Recall
- ⭐ F1 Score
- 📋 Classification Report
- 📊 Confusion Matrix

🏆 Results

The trained BERT model achieved:

- 🎯 Validation Accuracy: Over 99%
- ⭐ F1 Score: 0.9695

The high F1 score demonstrates a strong balance between precision and recall.

---

📊 Confusion Matrix

A confusion matrix is generated using Seaborn to analyze:

- ✅ Correctly classified Ham messages
- 🚨 Correctly classified Spam messages
- ❌ False Positives
- ❌ False Negatives

This helps identify where the model makes incorrect predictions.

---

🔍 Real-Time Prediction

A prediction function is created to classify new messages.

The system returns:

Prediction
Confidence

Example 1 🚨

Input:

Congratulations! You have won a $1000 gift card. Click here now.

Output:

Spam

Example 2 ✅

Input:

Hi John, the meeting has been moved to 3 PM.

Output:

Ham

---

🌐 Gradio Web Application

The project includes an interactive Gradio interface.

Users can enter a message into the text box and receive a prediction such as:

Spam (98.45%)

or

Ham (99.72%)

This makes the trained BERT model easier to use without writing Python code.

---

💾 Model Saving

The trained BERT model and tokenizer are saved locally.

model.save_pretrained("./spam_bert_model")
tokenizer.save_pretrained("./spam_bert_model")

The project also saves the best-performing model based on validation F1 score:

./best_spam_model

---

📁 Project Structure

📦 Spam-Email-Detection-BERT
│
├── 📓 Spam_Email_Detection_BERT.ipynb
├── 📄 spam.csv
├── 📁 best_spam_model/
├── 📁 spam_bert_model/
├── 📄 README.md
└── 📄 requirements.txt

---

⚙️ Installation

Install the required Python libraries:

pip install torch
pip install transformers
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install gradio

Or install everything using:

pip install torch transformers pandas numpy scikit-learn matplotlib seaborn gradio

---

▶️ How to Run

1️⃣ Clone the Repository

git clone <your-repository-url>

2️⃣ Navigate to the Project

cd Spam-Email-Detection-BERT

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Open the Notebook

Open:

Spam_Email_Detection_BERT.ipynb

using:

- 📓 Jupyter Notebook
- ☁️ Google Colab
- 💻 JupyterLab

5️⃣ Run the Notebook

Run the cells in order to:

1. Load the dataset
2. Preprocess the data
3. Tokenize the messages
4. Train BERT
5. Evaluate the model
6. Save the trained model
7. Launch the Gradio application

---

💡 Key Learning Outcomes

Through this project, I learned how to:

- 🐍 Work with Python for machine learning
- 🧹 Preprocess text data
- 🔤 Perform NLP tokenization
- 🤗 Use pretrained Transformer models
- 🧠 Fine-tune BERT for classification
- 🔥 Work with PyTorch
- 📊 Evaluate classification models
- 📈 Create confusion matrices
- 💾 Save and load trained models
- 🌐 Build interactive ML applications using Gradio
- 🚀 Deploy a deep learning model for real-time prediction

---

🔎 Error Analysis

The model produced a small number of incorrect predictions.

Some errors occurred with ambiguous messages, particularly:

- 📱 Very short messages
- 🔢 Messages containing phone numbers
- 😄 Jokes or unusual text
- 📝 Messages with limited contextual information

🔮 Possible Improvements

Future versions could improve the model by:

- 📚 Using a larger and more diverse dataset
- 🔤 Increasing the maximum sequence length
- 🎛️ Performing hyperparameter tuning
- ⚖️ Handling class imbalance more carefully
- 🧠 Experimenting with other Transformer models
- 📊 Performing detailed error analysis

---

🚀 Future Improvements

Some potential improvements include:

- 📏 Compare BERT with traditional ML models
- 🤖 Experiment with DistilBERT and RoBERTa
- 🎛️ Hyperparameter optimization
- 📊 Add ROC-AUC and Precision-Recall curves
- 🌐 Deploy the application online
- 📱 Create a mobile-friendly interface
- 🔄 Automate model retraining
- ☁️ Deploy using cloud services
- 📈 Monitor model performance over time

---

🏁 Conclusion

This project successfully demonstrates how BERT and Deep Learning can be used for automated spam detection.

The fine-tuned BERT model achieved over 99% validation accuracy and an F1 score of 0.9695, demonstrating strong classification performance.

The addition of a Gradio interface makes the model interactive and allows users to test messages in real time.

Overall, this project provides practical experience with:

NLP + Transformers + BERT + PyTorch + Machine Learning + Deployment 🚀

---

👨‍💻 Author

KARTHIKEYAN M

📌 Machine Learning | Deep Learning | NLP | Artificial Intelligence

---

⭐ Support

If you find this project useful, please consider giving the repository a ⭐ Star.

Your support is appreciated! ❤️

---

🚀 Project Type

Machine Learning | Deep Learning | NLP | BERT | Text Classification | Spam Detection | Transformers
