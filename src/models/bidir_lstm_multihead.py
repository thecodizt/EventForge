from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Attention, Concatenate

def build_model(input_shape, num_classes, model_params):
    lstm_units = model_params.get('lstm_units', 64)
    num_lstm_layers = model_params.get('num_lstm_layers', 1)
    attention = model_params.get('attention', False)

    inputs = []
    lstm_outputs = []

    # Define input layers for each input type
    for i in range(3):  # Assuming 3 inputs (event, agent, context)
        inputs.append(Input(shape=(input_shape[0], 1)))

    # Concatenate all inputs
    concat = Concatenate()(inputs)

    # Add multiple LSTM layers
    lstm_layer = concat
    for _ in range(num_lstm_layers):
        lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_layer)
        lstm_outputs.append(lstm_layer)

    # Apply attention mechanism if specified
    if attention and num_lstm_layers > 1:
        attention_layer = Attention()([lstm_outputs[-1], lstm_outputs[-2]])
    elif attention:
        attention_layer = Attention()(lstm_outputs[-1])
    else:
        attention_layer = lstm_outputs[-1]

    # Final LSTM layer
    lstm_out = LSTM(lstm_units, return_sequences=False)(attention_layer)

    # Output layers for each prediction task
    outputs = []
    for i in range(3):  # Assuming 3 outputs (event, agent, context)
        outputs.append(Dense(num_classes[i], activation='softmax', name=f'output_{i + 1}')(lstm_out))

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'] * 3)  # Accuracy metrics for each output

    return model
