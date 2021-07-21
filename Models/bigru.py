from keras import layers, models, optimizers

class BiGru:
    def create(input_size, word_index, embedding_matrix, embedding_dim, num_classes, learning_rate):
        input_layer = layers.Input((input_size, ))
        embedding_layer = layers.Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
        bigru_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)
        output_layer1 = layers.Dense(50, activation='relu')(bigru_layer)
        output_layer1 = layers.Dropout(0.5)(output_layer1)
        output_layer2 = layers.Dense(num_classes, activation='sigmoid')(output_layer1)
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['acc', 'mse'])
        return model