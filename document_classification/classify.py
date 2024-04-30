import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Initializing paragraphs and assigning their labels
paragraphs = []
labels = []

label_map = {
    'historical_paragraphs': 'historical',
    'medical_paragraphs': 'medical',
    'legal_paragraphs': 'legal'
}

# Reading file
current_label = None
with open('dataset/data.txt', 'r', encoding='utf-8') as file: # This reads from my directory, needs to specify path to dataset
    for line in file:
        line = line.strip()
        if line.endswith('['):
            # Section header
            current_label = label_map[line.split()[0]]
        elif line.startswith('"') and current_label:
            # Paragraph
            paragraph = line.strip('",')
            paragraphs.append(paragraph)
            labels.append(current_label)

# Tokenizing text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(paragraphs)
sequences = tokenizer.texts_to_sequences(paragraphs)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Converting labels to one-hot encoding
label_dict = {'historical': 0, 'medical': 1, 'legal': 2}
labels = [label_dict[label] for label in labels]
one_hot_labels = np.zeros((len(labels), len(label_dict)))
for i, label in enumerate(labels):
    one_hot_labels[i, label] = 1

# Splitting data so that 30% of the dataset is reserved for testing, the other 70% will be used to train the model
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, one_hot_labels, test_size=0.3, random_state=42)

# Model
model = Sequential([
    Embedding(vocab_size, 100),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_dict), activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test))

# Plotting
plt.figure(figsize=(7, 3))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

model.save('classifier_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)