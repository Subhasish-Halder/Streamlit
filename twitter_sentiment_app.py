import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = tf.keras.models.load_model('sentiment_model.h5')

# Load the tokenizer
tokenizer = Tokenizer(num_words=5000)  # You need to set the same value as used during training

# Define the maximum sequence length (you should use the same value used during training)
max_length = 100  # Adjust this value based on your training data

# Create a function to predict sentiment
def predict_sentiment(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, truncating='post')

    # Predict the sentiment
    prediction = model.predict(padded)

    return prediction

# Set up Streamlit app
st.title("Sentiment Analysis App")

# Text input for user
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input:
        # Call the prediction function
        prediction = predict_sentiment(user_input)

        # Determine the sentiment based on the prediction
        sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax()]

        # Display the result
        st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text for analysis.")
