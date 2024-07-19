import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Attention, Concatenate, Layer

class GraphAttentionLayer(Layer):
    def __init__(self, units):
        super(GraphAttentionLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 name='W')
        self.a = self.add_weight(shape=(2*self.units, 1),
                                 initializer='glorot_uniform',
                                 name='a')

    def call(self, inputs):
        h = tf.matmul(inputs, self.W)
        broadcast_shape = tf.shape(h)
        broadcast_shape = tf.concat([broadcast_shape[:-1], [1, broadcast_shape[-1]]], axis=0)
        repeated_h = tf.tile(tf.expand_dims(h, axis=2), [1, 1, tf.shape(h)[1], 1])
        repeated_h_t = tf.tile(tf.expand_dims(h, axis=1), [1, tf.shape(h)[1], 1, 1])
        concat = tf.concat([repeated_h, repeated_h_t], axis=-1)
        e = tf.nn.leaky_relu(tf.squeeze(tf.matmul(concat, self.a), axis=-1))
        attention = tf.nn.softmax(e, axis=-1)
        return tf.matmul(attention, h)

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

    # Apply GAT layer
    gat_output = GraphAttentionLayer(lstm_units)(concat)

    # Add multiple LSTM layers
    lstm_layer = gat_output
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