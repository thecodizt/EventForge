from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Concatenate

def build_model(input_shape, num_classes, model_params):
    num_filters = model_params.get('num_filters', 64)
    filter_sizes = model_params.get('filter_sizes', [3, 4, 5])

    inputs = []
    conv_outputs = []

    # Define input layers for each input type
    for i in range(3):  # Assuming 3 inputs (event, agent, context)
        inputs.append(Input(shape=(input_shape[0], 1)))

    # Convolutional layers for each input type
    for input_layer in inputs:
        conv_layers = []
        for filter_size in filter_sizes:
            conv = Conv1D(num_filters, filter_size, activation='relu')(input_layer)
            conv = GlobalMaxPooling1D()(conv)
            conv_layers.append(conv)
        conv_outputs.extend(conv_layers)

    # Concatenate convolutional outputs
    concat = Concatenate()(conv_outputs)

    # Dense layers for classification
    dense = Dense(128, activation='relu')(concat)
    outputs = []
    for i in range(3):  # Assuming 3 outputs (event, agent, context)
        outputs.append(Dense(num_classes[i], activation='softmax', name=f'output_{i + 1}')(dense))

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'] * 3)  # Accuracy metrics for each output

    return model
