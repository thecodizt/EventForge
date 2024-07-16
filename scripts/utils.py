import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import load_model

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU is available. Enabling GPU acceleration.")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    device = '/GPU:0'
else:
    print("No GPU available. Running on CPU.")
    device = '/CPU:0'

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['event_type', 'agent_id', 'context'])
    return df

def preprocess_data_label_encoder(df):
    e_event = LabelEncoder()
    e_agent = LabelEncoder()
    e_context = LabelEncoder()

    # Replace empty strings with a special token
    df['context'] = df['context'].replace('', '<EMPTY>')

    df['event_type_encoded'] = e_event.fit_transform(df['event_type'])
    df['agent_id_encoded'] = e_agent.fit_transform(df['agent_id'])
    df['context_encoded'] = e_context.fit_transform(df['context'])

    return df, e_event, e_agent, e_context

# Create sequences
def create_sequences(df, sequence_length):
    sequences = []
    targets = []

    for i in range(len(df) - sequence_length):
        seq = df.iloc[i:i+sequence_length]
        target = df.iloc[i+sequence_length]

        sequences.append(seq[['event_type_encoded', 'agent_id_encoded', 'context_encoded']].values)
        targets.append([target['event_type_encoded'], target['agent_id_encoded'], target['context_encoded']])

    return np.array(sequences), np.array(targets)

def predict_next_moves(model, initial_sequence, e_event, e_agent, e_context, n_predictions):
    with tf.device(device):
        predictions = []
        current_sequence = initial_sequence.copy()

        for _ in range(n_predictions):
            custom_event = current_sequence[:, 0].reshape(1, -1, 1)
            custom_agent = current_sequence[:, 1].reshape(1, -1, 1)
            custom_context = current_sequence[:, 2].reshape(1, -1, 1)

            prediction = model.predict([custom_event, custom_agent, custom_context])

            predicted_event = e_event.inverse_transform([np.argmax(prediction[0])])[0]
            predicted_agent = e_agent.inverse_transform([np.argmax(prediction[1])])[0]
            predicted_context = e_context.inverse_transform([np.argmax(prediction[2])])[0]

            predicted_context = '' if predicted_context == '<EMPTY>' else predicted_context

            predictions.append((predicted_event, predicted_agent, predicted_context))

            # Update the sequence for the next prediction
            new_row = np.array([[
                e_event.transform([predicted_event])[0],
                e_agent.transform([predicted_agent])[0],
                e_context.transform([predicted_context if predicted_context != '' else '<EMPTY>'])[0]
            ]])
            current_sequence = np.vstack((current_sequence[1:], new_row))

        return predictions

def simulate_predictions(data, model, e_event, e_agent, e_context, n = 15, k = 10):
    # Predict next moves for a random sequence from the dataset

    # Select a random sequence from the dataset
    random_index = random.randint(0, len(data) - 1)
    initial_sequence = data[random_index]

    predictions = predict_next_moves(model, initial_sequence, e_event, e_agent, e_context, n)

    print("Initial sequence:")
    for i in range(k):
        event = e_event.inverse_transform([initial_sequence[i, 0]])[0]
        agent = e_agent.inverse_transform([initial_sequence[i, 1]])[0]
        context = e_context.inverse_transform([initial_sequence[i, 2]])[0]
        context = '' if context == '<EMPTY>' else context
        print(f"{event}, {agent}, {context}")

    print("\nPredicted moves:")
    for i, (event, agent, context) in enumerate(predictions, 1):
        print(f"Move {i}: {event}, {agent}, {context}")
        
def train_and_test(X_train, X_test, y_train, y_test, e_event, e_agent, e_context, build_model, name, epochs, batch_size):
    with tf.device(device):
        # Prepare inputs and outputs
        X_train_event = X_train[:, :, 0].reshape(X_train.shape[0], X_train.shape[1], 1)
        X_train_agent = X_train[:, :, 1].reshape(X_train.shape[0], X_train.shape[1], 1)
        X_train_context = X_train[:, :, 2].reshape(X_train.shape[0], X_train.shape[1], 1)

        X_test_event = X_test[:, :, 0].reshape(X_test.shape[0], X_test.shape[1], 1)
        X_test_agent = X_test[:, :, 1].reshape(X_test.shape[0], X_test.shape[1], 1)
        X_test_context = X_test[:, :, 2].reshape(X_test.shape[0], X_test.shape[1], 1)

        y_train_event = to_categorical(y_train[:, 0], num_classes=len(e_event.classes_))
        y_train_agent = to_categorical(y_train[:, 1], num_classes=len(e_agent.classes_))
        y_train_context = to_categorical(y_train[:, 2], num_classes=len(e_context.classes_))

        y_test_event = to_categorical(y_test[:, 0], num_classes=len(e_event.classes_))
        y_test_agent = to_categorical(y_test[:, 1], num_classes=len(e_agent.classes_))
        y_test_context = to_categorical(y_test[:, 2], num_classes=len(e_context.classes_))

        # Build and train model
        model = build_model(X_train_event.shape[1:],
                            [len(e_event.classes_), len(e_agent.classes_), len(e_context.classes_)])

        model.fit([X_train_event, X_train_agent, X_train_context],
                    [y_train_event, y_train_agent, y_train_context],
                    validation_data=([X_test_event, X_test_agent, X_test_context],
                                    [y_test_event, y_test_agent, y_test_context]),
                    epochs=epochs, batch_size=batch_size)

        # Evaluate model
        results = model.evaluate(
                [X_test_event, X_test_agent, X_test_context],
                [y_test_event, y_test_agent, y_test_context]
            )
            
        # Print evaluation results
        print("Test Results:")
        for metric_name, value in zip(model.metrics_names, results):
                print(f"{metric_name}: {value:.4f}")
                
        model.save(f'../../../models/{name}.h5')
                
        return model
    
def simulate_predictions_from_pretrained(data, modelpath, e_event, e_agent, e_context, n = 15, k = 10):

    # Select a random sequence from the dataset
    random_index = random.randint(0, len(data) - 1)
    initial_sequence = data[random_index]
    
    model = load_model(modelpath)

    predictions = predict_next_moves(model, initial_sequence, e_event, e_agent, e_context, n)

    print("Initial sequence:")
    for i in range(k):
        event = e_event.inverse_transform([initial_sequence[i, 0]])[0]
        agent = e_agent.inverse_transform([initial_sequence[i, 1]])[0]
        context = e_context.inverse_transform([initial_sequence[i, 2]])[0]
        context = '' if context == '<EMPTY>' else context
        print(f"{event}, {agent}, {context}")

    print("\nPredicted moves:")
    for i, (event, agent, context) in enumerate(predictions, 1):
        print(f"Move {i}: {event}, {agent}, {context}")