import re
import random
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

# 1. Load data
df = pd.read_csv("chat.csv")
texts = df["text"].tolist()
y = df["label"].values

# Replies for each label: 0 greeting, 1 thanks, 2 help, 3 bye
replies = {
    0: ["Hello! Welcome. How can I help you?", "Hi there! How can I assist you?"],
    1: ["You're welcome!", "Happy to help!", "No problem, anytime!"],
    2: ["Sure! How can I assist you?", "Of course, how can I help you?"],
    3: ["Goodbye! Have a great day.", "See you soon!"],
}

# 2. Words -> numbers (bag of words)
def tokenize(t):
    return re.findall(r"[a-z']+", t.lower())      # lowercase, no punctuation

vocab = sorted({w for t in texts for w in tokenize(t)})

def encode(t):
    words = set(tokenize(t))                      # split once
    return [1 if w in words else 0 for w in vocab]

X = np.array([encode(t) for t in texts])

# 3. Model
model = Sequential([
    Input(shape=(len(vocab),)),
    Dense(16, activation="relu"),
    Dense(4, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# 4. Train
model.fit(X, y, epochs=150, verbose=0)
loss, acc = model.evaluate(X, y, verbose=0)
print(f"Training finished. Accuracy on training data: {acc:.2f}")

# 5. Chat
print("Chatbot ready. Type 'quit' to exit.")
while True:
    msg = input("You: ")
    if msg.lower().strip() == "quit":
        print("Bot: Goodbye!")
        break

    vec = np.array([encode(msg)])
    if vec.sum() == 0:                            # no known words
        print("Bot: Sorry, I didn't understand that.")
        continue

    probs = model.predict(vec, verbose=0)[0]
    label = int(np.argmax(probs))
    if probs[label] < 0.5:                        # low confidence
        print("Bot: Sorry, I didn't understand that.")
    else:
        print("Bot:", random.choice(replies[label]))