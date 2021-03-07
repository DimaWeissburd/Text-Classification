from keras import layers, models, optimizers

class CreateRnnGru:
    def create(word_index, embedding_matrix, learning_rate, num_classes):
        # Add an Input Layer
        input_layer = layers.Input((70, ))
        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
        # Add the GRU Layer
        lstm_layer = layers.GRU(100)(embedding_layer)
        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(num_classes, activation="sigmoid")(output_layer1)
        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        return model