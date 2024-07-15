from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate,Bidirectional, Attention

# Build model
def build_model(input_shape, num_classes):
    input_event = Input(shape=(input_shape[0], 1))
    input_agent = Input(shape=(input_shape[0], 1))
    input_context = Input(shape=(input_shape[0], 1))

    concat = Concatenate()([input_event, input_agent, input_context])

    lstm = Bidirectional(LSTM(64, return_sequences=True))(concat)
    attention = Attention()([lstm, lstm])
    lstm_out = LSTM(64, return_sequences=False)(attention)

    output_event = Dense(num_classes[0], activation='softmax', name='event_type')(lstm_out)
    output_agent = Dense(num_classes[1], activation='softmax', name='agent_id')(lstm_out)
    output_context = Dense(num_classes[2], activation='softmax', name='context')(lstm_out)

    model = Model(inputs=[input_event, input_agent, input_context],
                  outputs=[output_event, output_agent, output_context])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy','accuracy','accuracy'])

    return model