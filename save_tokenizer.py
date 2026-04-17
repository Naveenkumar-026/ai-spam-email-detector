import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("spam_dataset_cleaned.csv")  # Ensure this file exists in your directory

# Ensure all text values are strings and fill NaN with an empty string
df['cleaned_text'] = df['cleaned_text'].astype(str).fillna("")

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=5000)  # Use the same num_words as in training
tokenizer.fit_on_texts(df['cleaned_text'])  # Train tokenizer on cleaned text

# Save the tokenizer as a .pkl file
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

print("Tokenizer saved successfully as 'tokenizer.pkl'")
