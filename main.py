#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import pickle
#######################
# Step 1: Import Libraries and Load the Model
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
#######################

# Page configuration
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

#######################
# Sidebar
with st.sidebar:
    st.title('🏂 US Population Dashboard')
    
    project = ['Sentiment Analysis','Next word Prediction','Text Classification']
    
    selected_project= st.selectbox('Select a year', project)
    #df_selected_year = df_reshaped[df_reshaped.year == selected_year]
    #df_selected_year_sorted = df_selected_year.sort_values(by="population", ascending=False)

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)


#######################

# ---------------------------- Sentiment Analysis --------------------------------#
if selected_project == 'Sentiment Analysis':

    # Load the IMDB dataset word index
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}

    # Load the pre-trained model with ReLU activation
    model = load_model('simple_rnn/simple_rnn_imdb.h5')

    # Step 2: Helper Functions
    # Function to decode reviews
    def decode_review(encoded_review):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

    # Function to preprocess user input
    def preprocess_text(text):
        words = text.lower().split()
        encoded_review = [word_index.get(word, 2) + 3 for word in words]
        padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
        return padded_review

    # Streamlit app
    st.title('IMDB Movie Review Sentiment Analysis')
    st.write('Enter a movie review to classify it as positive or negative.')

    # User input
    user_input = st.text_area('Movie Review')

    if st.button('Classify'):

        preprocessed_input=preprocess_text(user_input)

        ## MAke prediction
        prediction=model.predict(preprocessed_input)
        sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]}')
    else:
        st.write('Please enter a movie review.')

#############################
#-----------------------------------------------------------------------------------------------#

#------------------------Next word prediction--------------------------------#
elif selected_project == 'Next word Prediction':
    
    # Load the LSTM Model
    model = load_model('lstm_rnn/next_word_lstm.h5')

    ## load the tokenizer
    with open('lstm_rnn/tokenizer.pickle','rb') as handle:
        tokenizer = pickle.load(handle)

    # Function to predict the next word
    def predict_next_word(model, tokenizer, text, max_sequence_len):
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        return None
    
    # streamlit app
    st.title("Next Word Prediction With LSTM And Early Stopping")
    input_text=st.text_input("Enter the sequence of Words","To be or not to")
    if st.button("Predict Next Word"):
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f'Next word: {next_word}')

#############################
#-----------------------------------------------------------------------------------------------#

#------------------------Text Classification--------------------------------#
elif selected_project == 'Text Classification': 
    st.write('Welcome')