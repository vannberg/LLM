import numpy as np
import os

# Vocabulary definition
common_words = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
]

digits = [str(i) for i in range(10)]
alphabet = [chr(i) for i in range(ord('A'), ord('Z')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)]
accessory_chars = ['\t', '\n', '!', '?', '.', ',', ':', ';', '(', ')', '[', ']', '{', '}', '"', "'", '`', '-', '_', '=', '+', '*', '/', '\\', '|', '&', '%', '$', '#', '@', '^', '~', '<', '>']

vocab = common_words + digits + alphabet + accessory_chars
vocab = vocab[:256]  # Truncate to exactly 256 items
vocab_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_vocab = {idx: word for word, idx in vocab_to_index.items()}

# Network parameters
vocab_size = 256  # Ensure this matches the vocabulary size
hidden_size = 3
learning_rate = 0.01

# Helper functions for saving and loading arrays
def save_array_to_file(array, filename):
    with open(filename, 'w') as file:
        for row in array:
            if array.ndim > 1:
                np.savetxt(file, row.reshape(1, -1), fmt='%.18e', delimiter=',')
            else:
                file.write(f"{row}\n")

def load_array_from_file(filename, shape):
    with open(filename, 'r') as file:
        lines = file.readlines()
        if len(shape) == 2:
            array = np.array([list(map(float, line.strip().split(','))) for line in lines])
            return array.reshape(shape)
        elif len(shape) == 1:
            return np.array([float(line.strip()) for line in lines])

# Sample text for training
sample_text = """Machine learning (ML) is a field of inquiry devoted to understanding and building methods that learn from data. It is seen as a subset of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to perform the task."""

# Tokenize the sample text
def tokenize(text):
    tokens = []
    for word in text.split():
        if word in vocab_to_index:
            tokens.append(vocab_to_index[word])
        else:
            tokens.extend([vocab_to_index[c] if c in vocab_to_index else vocab_to_index['?'] for c in word])
    return tokens

training_data = [tokenize(sample_text)]

# Activation and loss functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def cross_entropy_loss(predicted, actual):
    return -np.log(predicted[actual])

def cross_entropy_loss_derivative(predicted, actual):
    # dL/dY where Y is the output of softmax
    dL_dY = predicted.copy()
    dL_dY[actual] -= 1
    return dL_dY

# Forward pass
def forward_pass(input_vector, W_input_hidden, W_hidden_output, b_hidden, b_output):
    hidden = relu(np.dot(input_vector, W_input_hidden) + b_hidden)
    output = softmax(np.dot(hidden, W_hidden_output) + b_output)
    return hidden, output

# Backpropagation
def backward_pass(input_vector, hidden, output, actual_token):
    global W_input_hidden, W_hidden_output, b_hidden, b_output

    # Compute gradients
    dL_dY = cross_entropy_loss_derivative(output[0], actual_token)
    dY_dW_ho = hidden  # This should be (1, 3) for multiplication
    dY_db_o = 1  # Scalar since b_output is (1, 256)
    
    # Ensure shapes are correct for dot product
    dL_dW_ho = np.outer(dY_dW_ho[0], dL_dY)  # This will give us (3, 256) matching W_hidden_output shape
    dL_db_o = dL_dY  # This matches b_output's shape (1, 256) when we consider dL_dY as (256,)
    
    dY_dH = W_hidden_output.T
    dH_dZ = relu_derivative(np.dot(input_vector, W_input_hidden) + b_hidden)
    dZ_dW_ih = input_vector.T  # (1, 256) for input_vector
    dZ_db_h = 1  # Scalar since b_hidden is (1, 3)

    dL_dZ = dL_dY.dot(dY_dH) * dH_dZ[0]  # dL_dY is (256,), dY_dH is (256, 3), dH_dZ is (1, 3)
    dL_dW_ih = np.outer(dZ_dW_ih[0], dL_dZ)  # This will give us (256, 3) matching W_input_hidden shape
    dL_db_h = dL_dZ  # dL_dZ is (3,) matches b_hidden's shape (1, 3) when we consider all as scalar operations

    # Update weights and biases
    W_input_hidden -= learning_rate * dL_dW_ih
    W_hidden_output -= learning_rate * dL_dW_ho
    b_hidden -= learning_rate * dL_db_h.reshape(b_hidden.shape)
    b_output -= learning_rate * dL_db_o.reshape(b_output.shape)

# Check if weight files exist and load them if they do
weight_files = ["W_input_hidden.txt", "W_hidden_output.txt", "b_hidden.txt", "b_output.txt"]
if all(os.path.isfile(file) for file in weight_files):
    print("Loading existing weights...")
    W_input_hidden = load_array_from_file("W_input_hidden.txt", (vocab_size, hidden_size))
    W_hidden_output = load_array_from_file("W_hidden_output.txt", (hidden_size, vocab_size))
    b_hidden = load_array_from_file("b_hidden.txt", (1, hidden_size))
    b_output = load_array_from_file("b_output.txt", (1, vocab_size))
else:
    print("Initializing weights...")
    # Initialize weights and biases
    W_input_hidden = np.random.randn(vocab_size, hidden_size) / np.sqrt(vocab_size)  # He initialization
    W_hidden_output = np.random.randn(hidden_size, vocab_size) / np.sqrt(hidden_size)
    b_hidden = np.zeros((1, hidden_size))
    b_output = np.zeros((1, vocab_size))

# Training loop (remains unchanged)
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0
    for sequence in training_data:
        for i in range(len(sequence) - 1):
            input_vector = np.zeros((1, vocab_size))
            input_vector[0, sequence[i]] = 1  # One-hot encode the current word

            hidden, output = forward_pass(input_vector, W_input_hidden, W_hidden_output, b_hidden, b_output)
            loss = cross_entropy_loss(output[0], sequence[i + 1])
            total_loss += loss

            # Perform backpropagation
            backward_pass(input_vector, hidden, output, sequence[i + 1])

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss}")

# Save weights after training
save_array_to_file(W_input_hidden, "W_input_hidden.txt")
save_array_to_file(W_hidden_output, "W_hidden_output.txt")
save_array_to_file(b_hidden, "b_hidden.txt")
save_array_to_file(b_output, "b_output.txt")

# Prediction function (remains unchanged)
def predict_next_token(sequence, W_input_hidden, W_hidden_output, b_hidden, b_output):
    input_vector = np.zeros((1, vocab_size))
    input_vector[0, sequence[-1]] = 1  # One-hot encode the last token in the sequence
    _, output = forward_pass(input_vector, W_input_hidden, W_hidden_output, b_hidden, b_output)

    # Debug: Print probabilities
    print(f"Probabilities: {output[0]}")
    print(f"Top 5 predictions: {[index_to_vocab[i] for i in np.argsort(output[0])[-5:][::-1]]}")
    
    return np.argmax(output)

# Test prediction
test_text = "is seen as"
test_sequence = tokenize(test_text)
predicted_token_index = predict_next_token(test_sequence, W_input_hidden, W_hidden_output, b_hidden, b_output)
predicted_token = index_to_vocab[predicted_token_index]

print(f"Input sequence: '{test_text}'")
print(f"Predicted next token: {predicted_token}")
