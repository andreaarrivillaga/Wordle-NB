import numpy as np
import random
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load word list
with open("words2.csv") as f:
    word_list = [word.strip().lower() for word in f.readlines() if len(word.strip()) == 5]

print(f"Total valid 5-letter words: {len(word_list)}")

# Generate training data
def generate_training_data(word_list):
    data, labels = [], []
    for target_word in random.sample(word_list, 500):  # Randomly sample target words
        for word in word_list:
            feedback = []
            target_list = list(target_word)
            for i in range(5):
                if word[i] == target_word[i]:
                    feedback.append(2)  # Green
                    target_list[i] = None
                elif word[i] in target_list:
                    feedback.append(1)  # Yellow
                    target_list[target_list.index(word[i])] = None
                else:
                    feedback.append(0)  # Gray
            data.append(word)  # Each word gets one label
            labels.append(feedback)  # Append the feedback for the word
    return data, labels

# Vectorize words
def custom_vectorize(word_list):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    vectorized = []
    for word in word_list:
        word_vector = []
        for letter in word:
            letter_vector = [0] * 26
            letter_vector[alphabet.index(letter)] = 1
            word_vector.extend(letter_vector)  # Append the one-hot vector for each letter
        vectorized.append(word_vector)  # Resulting vector will be of size 130
    return np.array(vectorized)

# Generate training data
training_words, training_labels = generate_training_data(word_list)
X = custom_vectorize(training_words)  # Shape: (n_samples, 130)
y = np.array(training_labels).reshape(-1, 5)  # Shape: (n_samples, 5)

# Ensure dimensions match
print(f"Vectorized data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Flatten labels for MultinomialNB
y_flat = y.ravel()  # Flatten to single dimension
X_flat = np.repeat(X, 5, axis=0)  # Repeat features to match flattened labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}, {y_train.shape}")
print(f"Test set size: {X_test.shape}, {y_test.shape}")

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)
print("Naive Bayes model trained.")

# Save the model and vectorizer for frontend use
with open("wordle_nb_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("wordle_nb_vectorizer.pkl", "wb") as f:
    pickle.dump(custom_vectorize, f)
print("Model and vectorizer saved.")
