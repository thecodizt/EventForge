import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Attention, Concatenate, Layer, TimeDistributed, Reshape

class GraphAttentionLayer(Layer):
    def __init__(self, units, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 name='W')
        self.a = self.add_weight(shape=(2*self.units, 1),
                                 initializer='glorot_uniform',
                                 name='a')
        super(GraphAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        h = tf.matmul(inputs, self.W)
        
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        a_input = tf.concat([tf.tile(h, [1, 1, seq_len]), tf.tile(tf.expand_dims(h, axis=2), [1, 1, seq_len, 1])], axis=-1)
        a_input = tf.reshape(a_input, (-1, seq_len, seq_len, 2*self.units))
        
        e = tf.nn.leaky_relu(tf.reduce_sum(tf.matmul(a_input, self.a), axis=-1))
        
        attention = tf.nn.softmax(e, axis=-1)
        
        out = tf.matmul(attention, h)
        
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

class HierarchicalRNN(Layer):
    def __init__(self, low_level_units, high_level_units, **kwargs):
        super(HierarchicalRNN, self).__init__(**kwargs)
        self.low_level_rnn = LSTM(low_level_units, return_sequences=True)
        self.high_level_rnn = LSTM(high_level_units, return_sequences=True)
        self.low_level_units = low_level_units
        self.high_level_units = high_level_units

    def call(self, inputs):
        if inputs.shape.ndims == 4:
            inputs = tf.squeeze(inputs, axis=2)
        
        low_level_output = self.low_level_rnn(inputs)
        
        seq_len = tf.shape(inputs)[1]
        reshaped = tf.reshape(low_level_output, (-1, seq_len // 5, 5, low_level_output.shape[-1]))
        high_level_input = tf.reduce_mean(reshaped, axis=2)
        
        high_level_output = self.high_level_rnn(high_level_input)
        
        high_level_output_upsampled = tf.repeat(high_level_output, repeats=5, axis=1)
        
        return tf.concat([low_level_output, high_level_output_upsampled], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.low_level_units + self.high_level_units)

def build_model(input_shape, num_classes, model_params):
    lstm_units = model_params.get('lstm_units', 64)
    num_lstm_layers = model_params.get('num_lstm_layers', 1)
    attention = model_params.get('attention', False)
    gat_units = model_params.get('gat_units', 32)
    hrnn_low_units = model_params.get('hrnn_low_units', 32)
    hrnn_high_units = model_params.get('hrnn_high_units', 16)

    inputs = []
    gat_outputs = []

    for i in range(3):
        inputs.append(Input(shape=(input_shape[0], 1)))

    for inp in inputs:
        gat_output = TimeDistributed(GraphAttentionLayer(gat_units))(inp)
        gat_outputs.append(gat_output)

    concat = Concatenate()(gat_outputs)

    hrnn_output = HierarchicalRNN(hrnn_low_units, hrnn_high_units)(concat)

    lstm_outputs = [hrnn_output]
    for _ in range(num_lstm_layers):
        lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_outputs[-1])
        lstm_outputs.append(lstm_layer)

    if attention and num_lstm_layers > 1:
        attention_layer = Attention()([lstm_outputs[-1], lstm_outputs[-2]])
    elif attention:
        attention_layer = Attention()(lstm_outputs[-1])
    else:
        attention_layer = lstm_outputs[-1]

    lstm_out = LSTM(lstm_units, return_sequences=False)(attention_layer)

    outputs = []
    for i in range(3):
        outputs.append(Dense(num_classes[i], activation='softmax', name=f'output_{i + 1}')(lstm_out))

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'] * 3)

    return model
