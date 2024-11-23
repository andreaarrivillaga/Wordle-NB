import streamlit as st
import random
import pickle
import numpy as np

# Load the word list
with open("words2.csv") as f:
    word_list = [word.strip().lower() for word in f.readlines() if len(word.strip()) == 5]

print(f"Total valid 5-letter words: {len(word_list)}")


# Define the custom vectorize function
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


@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained Naive Bayes model and vectorizer."""
    with open("wordle_nb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("wordle_nb_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def give_feedback(guess, target_word):
    """Provide feedback for the guess."""
    feedback = []  # (color, letter)
    target_word_list = list(target_word)

    for i, char in enumerate(guess):
        if char == target_word[i]:
            feedback.append(("green", char))  # Correct position
            target_word_list[i] = None  # Remove matched letters
        elif char in target_word_list:
            feedback.append(("yellow", char))  # Correct letter, wrong position
            target_word_list[target_word_list.index(char)] = None
        else:
            feedback.append(("gray", char))  # Incorrect letter

    return feedback


def filter_valid_words(valid_words, guesses):
    """Filter valid words based on feedback from previous guesses."""
    excluded_letters = set()

    for guess, feedback in guesses:
        for i, (color, letter) in enumerate(feedback):
            if color == "green":
                valid_words = [word for word in valid_words if word[i] == letter]
            elif color == "yellow":
                valid_words = [word for word in valid_words if letter in word and word[i] != letter]
            elif color == "gray":
                if not any(fb[0] in {"green", "yellow"} and fb[1] == letter for fb in feedback):
                    excluded_letters.add(letter)

    valid_words = [word for word in valid_words if all(letter not in word for letter in excluded_letters)]

    return valid_words


def suggest_top_words(model, vectorizer, valid_words, guesses, top_n=3):
    """
    Suggest the top `top_n` words using the Naive Bayes model and filtered valid words.
    """
    valid_words = filter_valid_words(valid_words, guesses)

    if not valid_words:
        return [("NO SUGGESTIONS", 0.0)] * top_n

    try:
        # Vectorize the valid words
        vectorized = vectorizer(valid_words)

        # Model inference: predict probabilities
        probabilities = model.predict_proba(vectorized).max(axis=1)

        # Get the top suggestions
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        suggestions = [(valid_words[i], probabilities[i]) for i in top_indices]

        # Pad with "NO SUGGESTIONS" if fewer than `top_n` words are available
        while len(suggestions) < top_n:
            suggestions.append(("NO SUGGESTIONS", 0.0))

        return suggestions

    except Exception as e:
        st.error(f"Error during word suggestion: {e}")
        return [("NO SUGGESTIONS", 0.0)] * top_n


def reset_game():
    """Initialize or reset the game state."""
    if "game_initialized" not in st.session_state:
        st.session_state.attempts = 6
        st.session_state.guesses = []
        st.session_state.current_guess = ""
        st.session_state.chosen_word = random.choice(word_list)
        st.session_state.valid_words = word_list.copy()
        st.session_state.suggested_words = []
        st.session_state.game_initialized = True


# Streamlit UI
st.title("Wordle Game - Naive Bayes Suggestions")
st.write("Enter a 5-letter word and get feedback with suggestions!")

# Load the model and vectorizer
model, vectorizer = load_model_and_vectorizer()
reset_game()

# Text input for guesses
guess = st.text_input("Type your guess (5 letters):").strip().lower()

if guess and len(guess) == 5:
    if guess not in word_list:
        st.warning("Invalid word. Please enter a valid 5-letter word.")
    else:
        feedback = give_feedback(guess, st.session_state.chosen_word)
        st.session_state.guesses.append((guess, feedback))
        st.session_state.attempts -= 1
        st.session_state.valid_words = [word for word in st.session_state.valid_words if word != guess]

        top_suggestions = suggest_top_words(model, custom_vectorize, st.session_state.valid_words, st.session_state.guesses)

        st.write("Top suggestions:")
        for idx, (suggestion, prob) in enumerate(top_suggestions, start=1):
            st.write(f"{idx}. {suggestion.upper()} (Probability: {prob:.4f})")

# Display the game board
rows, cols = 6, 5
for i in range(rows):
    cols_display = st.columns(cols)
    if i < len(st.session_state.guesses):
        guess, feedback = st.session_state.guesses[i]
        for j in range(cols):
            color, letter = feedback[j]
            background_color = {"green": "#00FF00", "yellow": "#FFFF00", "gray": "#D3D3D3"}.get(color, "#FFFFFF")
            cols_display[j].markdown(
                f"<div style='text-align:center; background-color:{background_color}; color:black; border:1px solid black;'>{letter.upper()}</div>",
                unsafe_allow_html=True,
            )
    else:
        for j in range(cols):
            cols_display[j].markdown(
                f"<div style='text-align:center; color:black; border:1px solid black;'>&nbsp;</div>",
                unsafe_allow_html=True,
            )

# Remaining attempts
st.write(f"Remaining Attempts: {st.session_state.attempts}")

# Check for win/loss condition
if len(st.session_state.guesses) > 0 and all(color == "green" for color, _ in st.session_state.guesses[-1][1]):
    st.success("Congratulations! You guessed the word!")
    st.session_state.game_initialized = False
elif st.session_state.attempts == 0:
    st.error(f"Game Over! The correct word was: {st.session_state.chosen_word.upper()}")
    st.session_state.game_initialized = False
