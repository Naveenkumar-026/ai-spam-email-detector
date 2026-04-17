# ==========================================
# Email Spam Detection using CNN + LSTM Model
# ==========================================

# ========== Import Necessary Libraries ==========
import os
import email
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet  # Add wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler

# ========== Step 1: Load the Dataset ==========
# Define paths to the directories
base_dir = 'data/'
easy_ham_path = os.path.join(base_dir, 'easy_ham')
hard_ham_path = os.path.join(base_dir, 'hard_ham')
spam_path = os.path.join(base_dir, 'spam')
spam_2_path = os.path.join(base_dir, 'spam_2')

# Initialize lists to hold email contents and labels
emails = []
labels = []
subjects = []
senders = []

# Modify load_emails_from_directory function above to:
def load_emails_from_directory(directory, label):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='latin-1') as file:
                try:
                    msg = email.message_from_file(file)
                    subject = msg.get("Subject", "")
                    sender = msg.get("From", "")

                    if msg.is_multipart():
                        content = ''
                        for part in msg.get_payload():
                            if part.get_content_type() == 'text/plain':
                                content += part.get_payload()
                    else:
                        content = msg.get_payload()

                    emails.append(content)
                    subjects.append(subject)
                    senders.append(sender)
                    labels.append(label)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")


# Load ham and spam emails
load_emails_from_directory(easy_ham_path, 0)   # 0 for ham
load_emails_from_directory(hard_ham_path, 0)   # 0 for ham
load_emails_from_directory(spam_path, 1)       # 1 for spam
load_emails_from_directory(spam_2_path, 1)     # 1 for spam

# Create a DataFrame
data = pd.DataFrame({'subject': subjects, 'sender': senders, 'body': emails, 'label': labels})
data["full_text"] = data["subject"] + " " + data["sender"] + " " + data["body"]

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Display the first few rows
print("Loaded Data:")
print(data.head())

# ========== Step 1.5: Load Additional Emails from CSV ==========
csv_data = pd.read_csv("real_emails_dataset.csv")

# Fill missing columns to avoid errors
csv_data["subject"] = csv_data["subject"].fillna("")
csv_data["sender"] = csv_data["sender"].fillna("")
csv_data["body"] = csv_data["body"].fillna("")

# Create full_text field
csv_data["full_text"] = csv_data["subject"] + " " + csv_data["sender"] + " " + csv_data["body"]

# Match original structure: append CSV data to existing DataFrame
original_data = pd.DataFrame({
    "subject": subjects,
    "sender": senders,
    "body": emails,
    "label": labels
})
original_data["full_text"] = original_data["subject"] + " " + original_data["sender"] + " " + original_data["body"]

# Combine both
data = pd.concat([original_data, csv_data], ignore_index=True)

# ========== Step 1.5: Load Additional Emails from CSV ==========
csv_data = pd.read_csv("realistic_spam_ham_dataset.csv")

# Ensure no nulls
csv_data["subject"] = csv_data["subject"].fillna("")
csv_data["sender"] = csv_data["sender"].fillna("")
csv_data["body"] = csv_data["body"].fillna("")

# Add a combined field
csv_data["full_text"] = csv_data["subject"] + " " + csv_data["sender"] + " " + csv_data["body"]

# Ensure correct label column is integer
csv_data["label"] = csv_data["label"].astype(int)

# Append to original dataframe
data = pd.concat([data, csv_data], ignore_index=True)

# Shuffle again after combining
data = data.sample(frac=1).reset_index(drop=True)

print("Combined with CSV data. Final dataset size:", len(data))

# ========== Step 1.6: Load Real-World Ham Dataset ==========

real_ham_data = pd.read_csv("real_world_ham_dataset.csv")

# Ensure essential columns are filled
real_ham_data["subject"] = real_ham_data["subject"].fillna("")
real_ham_data["sender"] = real_ham_data["sender"].fillna("")
real_ham_data["body"] = real_ham_data["body"].fillna("")

# Create unified full_text column
real_ham_data["full_text"] = (
    real_ham_data["subject"] + " " +
    real_ham_data["sender"] + " " +
    real_ham_data["body"]
)

# Label this dataset as clean (ham)
real_ham_data["label"] = 0

# Append to existing data
data = pd.concat([data, real_ham_data], ignore_index=True)

print(f"Real-world clean emails added. Dataset size: {len(data)}")


# ========== Step 2: Text Cleaning and Preprocessing ==========
# Initialize NLTK tools
nltk.download('wordnet')  # Ensure wordnet is available
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = word_tokenize(text)  # Tokenization
    words = [ps.stem(word) for word in words if word not in stop_words]  # Stemming and Stopwords removal
    return ' '.join(words)

# Apply preprocessing to the text column
data['cleaned_text'] = data['full_text'].apply(preprocess)

# Display the first few rows of cleaned data
print("\nCleaned Data:")
print(data[['full_text', 'cleaned_text', 'label']].head())

# Save the cleaned dataset as a CSV file (optional)
data.to_csv('spam_dataset_cleaned.csv', index=False)

# ========== Step 3: Feature Extraction using Word Embeddings ==========
def synonym_replacement(text, n=2):
    if not isinstance(text, str) or not text.strip():  # Ensure it's a valid string
        return text

    words = text.split()
    new_words = words.copy()

    for _ in range(n):
        if len(words) == 0:
            break  # Avoid errors on empty text

        word_idx = np.random.randint(0, len(words))
        synonyms = set()

        for syn in wordnet.synsets(words[word_idx]):
            for l in syn.lemmas():
                synonyms.add(l.name())

        if synonyms:
            new_word = list(synonyms)[0]
            new_words[word_idx] = new_word

    return ' '.join(new_words)


data['augmented_text'] = data['cleaned_text'].apply(lambda x: synonym_replacement(x))

print("Augmented dataset size:", len(data))

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['cleaned_text'])

# Convert text to sequences
X = tokenizer.texts_to_sequences(data['cleaned_text'])

# Pad sequences to ensure equal length
X = pad_sequences(X, maxlen=500)

y = data['label']

# ========== Step 4: Split Data into Training and Testing Sets ==========
# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ========== Step 5: Handle Class Imbalance ==========
# Calculate class weights to handle class imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: class_weights[0], 1: class_weights[1]}
print("\nClass Weights:", class_weights)

# ========== Step 6: Build the CNN + LSTM Hybrid Model ==========
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=500),

    Conv1D(64, kernel_size=5, use_bias=False, kernel_regularizer=L2(0.01)),  # Remove bias for BatchNorm
    BatchNormalization(),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),  # Slightly reduce dropout to 0.4


    Bidirectional(LSTM(100, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)),
    BatchNormalization(),  # Add Batch Normalization
    Dropout(0.5),

    Bidirectional(LSTM(50, dropout=0.5, recurrent_dropout=0.5)),
    Dropout(0.5),

    Dense(1, activation='sigmoid', kernel_regularizer=L2(0.005))  # Reduced L2 Regularization

])

# Compile the Model
optimizer = Adam(learning_rate=1e-4, epsilon=1e-8)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# Model Summary
print("\nModel Summary:")
model.summary()

# ========== Step 7: Train the Model ==========
# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6, verbose=1)

# Cyclical Learning Rate Scheduler (CLR)

def clr_schedule(epoch):
    base_lr = 1e-5
    max_lr = 3e-4
    step_size = 4  # Adjust learning rate every 4 epochs
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    new_lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return new_lr

clr = LearningRateScheduler(clr_schedule)

# Train the Model with Cyclical Learning Rate
history = model.fit(
    X_train, y_train, 
    epochs=12,  # Increase epochs slightly
    batch_size=64, 
    validation_data=(X_test, y_test), 
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr, clr]  # Add CLR
)

# ========== Step 8: Evaluate the Model ==========
# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print('\nTest Accuracy:', accuracy)

# ========== Step 9: Make Predictions and Evaluate ==========
# Make Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).flatten()

# Accuracy, Precision, Recall, F1-Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print("\nDetailed Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# ========== Step 10 (Updated): Save the Fine-Tuned Model ==========
# Save the fine-tuned model
model.save('fine_tuned_spam_detection_model.keras')
print("\nFine-Tuned Model saved successfully as 'fine_tuned_spam_detection_model.keras'")

# ========== Step 11: Visualize Training History ==========
import matplotlib.pyplot as plt

# Plot Training & Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Training & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

