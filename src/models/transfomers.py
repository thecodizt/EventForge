from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def build_model(input_shape, num_classes, model_params):
    d_model = model_params.get('d_model', 64)
    num_heads = model_params.get('num_heads', 4)
    ff_dim = model_params.get('ff_dim', 128)
    dropout_rate = model_params.get('dropout_rate', 0.1)

    inputs = []

    # Define input layers for each input type
    for i in range(3):  # Assuming 3 inputs (event, agent, context)
        inputs.append(Input(shape=(input_shape[0], 1)))

    # Concatenate all inputs
    concat = Concatenate()(inputs)

    # Apply MultiHeadAttention over the concatenated inputs
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(concat, concat, concat)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + concat)
    attention_output = Dropout(dropout_rate)(attention_output)

    # Add GlobalAveragePooling1D to reduce the sequence dimension
    pooled = GlobalAveragePooling1D()(attention_output)

    # Dense layers for classification
    dense = Dense(ff_dim, activation='relu')(pooled)
    outputs = []
    for i in range(3):  # Assuming 3 outputs (event, agent, context)
        outputs.append(Dense(num_classes[i], activation='softmax', name=f'output_{i + 1}')(dense))

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss=['categorical_crossentropy'] * 3,
                  metrics=['accuracy'] * 3)  # Specify metrics for each output

    return model