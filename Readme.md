# 🧠 Emotion Text Classifier

A simple machine learning project that classifies text into 6 emotions based on user input. It uses classical ML (TF-IDF + Logistic Regression) and a user-friendly Streamlit interface.

---

## 🎯 Emotions

The model detects the following emotions:
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😐 Neutral *(mapped from 'love' in original dataset)*
- 😨 Fear
- 😲 Surprise

---

## 🏗️ Project Structure

my-ml-project/
├── data/                  # CSVs from training (auto-generated)
├── model.py               # Script to train + save model
├── app.py                 # Streamlit app for real-time predictions
├── model.joblib           # Trained Logistic Regression model
├── vectorizer.joblib      # Saved TF-IDF vectorizer
├── requirements.txt       # All required dependencies
└── README.md              # You’re reading it!
---

## ⚙️ How to Set Up

### 1. Clone and enter the project

```bash
git clone https://github.com/yourusername/my-ml-project.git
cd my-ml-project
```

### 2. Create and activate a virtual environment
```
python -m venv ml-env
source ml-env/bin/activate  # or ml-env\Scripts\activate on Windows
```
### 3. Install dependencies 
```
pip install -r requirements.txt
```
## 🧪 Train the Model (Optional)

If you want to retrain the model from scratch:
python model.py

This will:
    •    Load Hugging Face emotion dataset
    •    Map 6 emotions 
    •    Train and evaluate model
    •    Save model.joblib, vectorizer.joblib, and CSVs to data/

## 🚀 Run the Streamlit App
```
streamlit run app.py
```
Then go to the browser and test it live!

## 🙏 Acknowledgements
    •    Hugging Face Datasets
    •    Scikit-learn
    •    Streamlit
