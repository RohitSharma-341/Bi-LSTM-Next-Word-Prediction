import streamlit as st
import tensorflow as tf
import numpy as np
import re
import json
import matplotlib.pyplot as plt
from collections import Counter
import random

# Set page config at the very beginning
st.set_page_config(page_title="Next Word Prediction", page_icon="📚", layout="wide")

# Custom model loading function
@st.cache_resource
def load_model():
    def remove_time_major(config):
        if isinstance(config, dict):
            return {k: remove_time_major(v) for k, v in config.items() if k != 'time_major'}
        elif isinstance(config, list):
            return [remove_time_major(item) for item in config]
        else:
            return config

    with tf.keras.utils.custom_object_scope({'TFBertMainLayer': tf.keras.layers.Layer}):
        model = tf.keras.models.load_model('model_bilstm.h5', compile=False)
        config = json.loads(model.to_json())
        cleaned_config = remove_time_major(config)
        new_model = tf.keras.models.model_from_json(json.dumps(cleaned_config))
        new_model.set_weights(model.get_weights())
    return new_model

model = load_model()

# Load the text of "The Metamorphosis"
@st.cache_data
def load_text():
    with open('metamorphosis_clean.txt', 'r', encoding='utf-8') as f:
        return f.read()

text = load_text()

# Create vocabulary
@st.cache_data
def create_vocabulary(text):
    words = re.findall(r'\w+', text.lower())
    word_counts = Counter(words)
    vocab = ['<UNK>'] + sorted(word_counts, key=word_counts.get, reverse=True)
    return {word: i for i, word in enumerate(vocab)}

vocab = create_vocabulary(text)
vocab_size = len(vocab)
inv_vocab = {v: k for k, v in vocab.items()}

# Tokenization function
def tokenize(text):
    return [vocab.get(word, vocab['<UNK>']) for word in re.findall(r'\w+', text.lower())]

# Function to predict the next word
def predict_next_word(model, text):
    sequence = tokenize(text)
    sequence = sequence[-50:]  # Use the last 50 words
    sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=model.input_shape[1], padding='pre')

    predicted = model.predict(sequence, verbose=0)
    predicted_word_index = np.argmax(predicted)

    return inv_vocab[predicted_word_index]

# Custom CSS with improved design
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #f3e7e9 0%, #e3eeff 99%, #e3eeff 100%);
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.8);
    }
    h1 {
        color: #2C3E50;
        text-shadow: 2px 2px 4px #ffffff;
        font-family: 'Georgia', serif;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.8);
        border: 2px solid #3498DB;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
    footer {
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("Next Word Prediction")

st.markdown("""
Welcome to the Next Word Prediction app based on Franz Kafka's "The Metamorphosis"! 
This AI uses Bi-directional LSTM to predict what word comes next in Gregor Samsa's surreal journey.

📚 Enter a phrase from the novel, and let's see what comes next!
""")

# Text input with a random placeholder
kafka_phrases = [
    "Gregor Samsa woke up",
    "As Gregor Samsa awoke one morning",
    "One morning, when Gregor Samsa",
    "He lay on his armour-like back",
]
placeholder = random.choice(kafka_phrases)
text_input = st.text_input("Enter some text from 'The Metamorphosis':", placeholder)

# Prediction
if st.button("Predict Next Word"):
    with st.spinner("Predicting..."):
        predicted_word = predict_next_word(model, text_input)

    st.success(f"The predicted next word is: **{predicted_word}**")

    # Visualize the prediction with improved design
    st.markdown("### Visualization")
    words = re.findall(r'\w+', text_input.lower()) + [predicted_word]
    colors = ['#AED6F1'] * len(words[:-1]) + ['#F1948A']

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(range(len(words)), [1] * len(words), color=colors)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=12)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.title("Word Sequence", fontsize=16)
    plt.tight_layout()

    st.pyplot(fig)

# Add some interactivity
st.sidebar.header("About the Model")
st.sidebar.markdown("""
This model uses Bi-LSTM (Bidirectional Long Short-Term Memory) 
trained on the text of Franz Kafka's "The Metamorphosis". It learns patterns 
in the narrative to predict the most likely next word in the sequence.

**Kafka's Opening Line:**
> "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect."
""")

# Add a feature to generate a random sentence
if st.sidebar.button("Generate a Random Sentence"):
    current_text = random.choice(["Gregor Samsa", "The metamorphosis", "One morning"])
    for _ in range(15):  # Generate 15 words for a longer sentence
        next_word = predict_next_word(model, current_text)
        current_text += f" {next_word}"
    st.sidebar.markdown(f"**Generated sentence:** {current_text}")

# Add a quiz feature
st.markdown("---")
st.subheader("Test Your Kafka Knowledge")
quiz_questions = [
    {"question": "What does Gregor Samsa transform into?", "answer": "insect"},
    {"question": "What is Gregor's profession before his transformation?", "answer": "salesman"},
    {"question": "What is the name of Gregor's sister?", "answer": "Grete"},
]
question = random.choice(quiz_questions)
user_answer = st.text_input(question["question"])
if st.button("Check Answer"):
    if user_answer.lower() == question["answer"]:
        st.success("Correct! You know your Kafka!")
    else:
        st.error(f"Not quite. The correct answer is '{question['answer']}'.")

st.markdown("---")
st.markdown("Made by RS", unsafe_allow_html=True)