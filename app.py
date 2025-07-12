import streamlit as st
import joblib
import numpy as np

# ✅ Custom labels from your model
LABEL_NAMES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ✅ Emoji mappings
EMOTION_EMOJIS = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Neutral": "😐",
    "Fear": "😨",
    "Surprise": "😲"
}

# ✅ Load model + vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_model()

# ✅ Streamlit UI
st.title("Emotion Classifier 🧠")
st.markdown("Enter a sentence and I'll detect its emotion!")

user_input = st.text_area("💬 Your Text:")

if user_input.strip():
    vec_input = vectorizer.transform([user_input])
    pred_proba = model.predict_proba(vec_input)[0]
    pred_index = np.argmax(pred_proba)

    emotion = LABEL_NAMES[pred_index]
    confidence = pred_proba[pred_index] * 100
    emoji = EMOTION_EMOJIS.get(emotion, "❓")

    st.markdown(f"### 🎯 Predicted Emotion: **{emotion}** {emoji}")
    st.write(f"🧪 **Confidence:** {confidence:.2f}%")

    # Show all probabilities
    st.subheader("📊 Emotion Probabilities")
    for i, label in enumerate(LABEL_NAMES):
        st.write(f"{label} {EMOTION_EMOJIS.get(label, '')}: {pred_proba[i]*100:.2f}%")
