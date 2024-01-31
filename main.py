import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

'''  
######################################################################################
'''


data = pd.read_excel("Dry_Bean_Dataset.xlsx")

# fix missing values
bom_missing = data[data['Class'] == 'BOMBAY']
bom_missing['MinorAxisLength'] = bom_missing['MinorAxisLength'].fillna(bom_missing['MinorAxisLength'].mean())
data['MinorAxisLength'][3] = bom_missing['MinorAxisLength'][3]


'''  
######################################################################################
'''
# User Input
num_hidden_layers = 1  # int(input("Enter number of hidden layers: "))
neurons_in_hidden_layers = [3]  # [int(input(f"Enter number of neurons in hidden layer {i + 1}: ")) for i in range(num_hidden_layers)]
learning_rate = 0.0001 # float(input("Enter learning rate (eta): "))
num_epochs = 2000  # int(input("Enter number of epochs (m): "))
add_bias = True  # input("Add bias? (yes/no): ").lower() == 'yes'
activation_function = 'tanh'  # input("Choose activation function (sigmoid/tanh): ").lower()
classes = data['Class'].unique()

'''  
######################################################################################
'''


def sliced_data():
    train = pd.DataFrame()
    test = pd.DataFrame()

    for c in classes:
        class1_train = data[data['Class'] == c].sample(n=30, random_state=42)
        class1_test = data[data['Class'] == c].drop(class1_train.index).sample(n=20, random_state=42)
        train = pd.concat([train, class1_train])
        test = pd.concat([test, class1_test])

    train = train.sample(frac=1, random_state=42)
    test = test.sample(frac=1, random_state=42)

    train = train.to_numpy()
    test = test.to_numpy()

    x_train = train[:, :-1]
    y_train = train[:, -1]

    for i in range(len(y_train)):
        if y_train[i] == classes[0]:
            y_train[i] = 0
        elif y_train[i] == classes[1]:
            y_train[i] = 1
        else:
            y_train[i] = 2

    x_test = test[:, :-1]
    y_test = test[:, -1]

    for i in range(len(y_test)):
        if y_test[i] == classes[0]:
            y_test[i] = 0
        elif y_test[i] == classes[1]:
            y_test[i] = 1
        else:
            y_test[i] = 2


    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')


    # Scale
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

train_data, test_data, train_labels, test_labels = sliced_data()

# Initialize
num_features = len(train_data[0])
num_classes = len(classes)
np.random.seed(42)

# Initialize weights and biases
weights = [np.ones((neurons_in_hidden_layers[0], num_features))]
biases = [np.zeros((neurons_in_hidden_layers[0], 1))]
for i in range(1, num_hidden_layers):
    weights.append(np.ones((neurons_in_hidden_layers[i], neurons_in_hidden_layers[i - 1])))
    biases.append(np.zeros((neurons_in_hidden_layers[i], 1)))
weights.append(np.ones((num_classes, neurons_in_hidden_layers[-1])))
biases.append(np.zeros((num_classes, 1)))


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh_derivative(x):
    return 1 - np.tanh(x)**2


def tanh(x):
    return np.tanh(x)


def activate(x):
    return sigmoid(x) if activation_function == 'sigmoid' else tanh(x)


def deactivate(x):
    return sigmoid_derivative(x) if activation_function == 'sigmoid' else tanh_derivative(x)


# Forward propagation
def forward_propagation(input_data):
    layer_outputs = [input_data.reshape((5,1))]
    for i in range(num_hidden_layers +1):
        layer_inputs = np.dot(weights[i], layer_outputs[i]) + biases[i]
        layer_outputs.append(activate(layer_inputs))
    return layer_outputs


# Backward propagation
def backward_propagation(outputs, lables):
    errors = [lables - outputs[-1]]
    for i in range(num_hidden_layers, 0, -1):
        errors.insert(0, np.dot(weights[i].T, errors[0]) * deactivate(outputs[i]))
    return errors


# Update weights and biases
def update_weights_and_biases(errors, layer_outputs):
    for i in range(num_hidden_layers + 1):
        weights[i] = weights[i] - learning_rate * np.dot(errors[i], layer_outputs[i].T)
        biases[i] = biases[i] - learning_rate * np.sum(errors[i], axis=1, keepdims=True)


# Training
for epoch in range(num_epochs):
    correct_train_predictions = 0

    for i in range(len(train_data)):

        layer_outputs = forward_propagation(train_data[i])

        errors = backward_propagation(layer_outputs, train_labels[i])

        update_weights_and_biases(errors, layer_outputs)

        # Calculate training accuracy
        predicted_class = np.argmax(layer_outputs[-1])
        # true_label = label_encoder.transform([train_labels[i]])[0]
        if predicted_class == train_labels[i]:
            correct_train_predictions += 1

    train_accuracy = correct_train_predictions / len(train_data)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {train_accuracy:.4f}")


# Testing
predictions = []
true_labels = []
for i in range(len(test_data)):
    # Forward propagation
    layer_outputs = forward_propagation(test_data[i])[-1]

    # Get predicted class
    predicted_class = np.argmax(layer_outputs)
    predictions.append(predicted_class)

    true_labels.append(test_labels[i])

# Evaluate
conf_matrix = confusion_matrix(true_labels, predictions)
accuracy = accuracy_score(true_labels, predictions)

# Display results
print("Confusion Matrix:")
print(conf_matrix)
print("\nOverall Accuracy:", accuracy)