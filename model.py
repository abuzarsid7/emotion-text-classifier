print("🟢 model.py is running")
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Step 1: Load dataset
dataset = load_dataset("emotion")
train_data = dataset["train"]
test_data = dataset["test"]

# Step 2: Convert to DataFrame
df = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

# Step 3: Replace label 2 (love) → Neutral
label_map = {
    0: "Sad",
    1: "Happy",
    2: "Neutral",   
    3: "Angry",
    4: "Fear",
    5: "Surprise"
}

df["label_text"] = df["label"].map(label_map)
df_test["label_text"] = df_test["label"].map(label_map)

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(df["text"])
X_test = vectorizer.transform(df_test["text"])

# Step 5: Labels
y_train = df["label_text"]
y_test = df_test["label_text"]

# Step 6: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
unique_labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")

# Step 8: Save dataset
project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)
df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)
print("✅ CSVs saved to:", data_dir)

# Classification Report
print(classification_report(y_test, y_pred, target_names=unique_labels))

# Step 9: Save model
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

# Show confusion matrix
plt.show()
