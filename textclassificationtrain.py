import os
import pandas, numpy
from sklearn import model_selection, preprocessing
from keras.preprocessing import text, sequence
from keras import optimizers, backend
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from Models.bigru import BiGru
from Models.cnn import Cnn

train_path = 'Data/Train.txt'
num_epochs = 20
learning_rate = 0.001
batch_size = 10

class TextClassificationTrain:
    def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, num_epochs, batch_size, save_file_name):
        checkpoint = ModelCheckpoint(save_file_name, monitor='loss', verbose=1,
            save_weights_only = False, save_best_only=False, mode='auto', save_freq='epoch')
        classifier.fit(feature_vector_train, label,
                        epochs=num_epochs,
                        validation_data=(feature_vector_valid, valid_y),
                        shuffle = True,
                        batch_size = batch_size,
                        verbose = 1,
                        callbacks=[checkpoint])

    def run_training(model, path, train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, embedding_dim, num_classes, num_epochs, batch_size, learning_rate):
        if os.path.exists(path):
            classifier = load_model(path)
            backend.set_value(classifier.optimizer.lr, learning_rate)
        else:
            classifier = model.create(len(train_seq_x[0]), word_index, embedding_matrix, embedding_dim, num_classes, learning_rate)
        TextClassificationTrain.train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, num_epochs, batch_size, path)

    def load_dataset(path):
        data = open(path, encoding='utf-8').read()
        texts, labels = [], []
        for i, line in enumerate(data.split('\n')):
            content = line.split('\t')
            texts.append(content[0])
            labels.append(content[1])
        return texts, labels

    def save_classes(path, classes):
        fc = open(path, 'w', encoding='utf-8')
        for index, type in enumerate(classes):
            if index == len(classes) - 1:
                fc.write(type)
            else:
                fc.write(type + '\n')
        fc.close()

    def create_embedding_matrix(word_index):
        embeddings_index = {}
        for i, line in enumerate(open('Data/glove.42B.300d.txt', 'r', encoding='utf-8', newline='\n', errors='ignore')):
            values = line.split()
            embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
        embedding_dim = len(embeddings_index.get(next(iter(embeddings_index))))
        embedding_matrix = numpy.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_dim, embedding_matrix

    def prepare_data():
        texts, labels = TextClassificationTrain.load_dataset(train_path)
        trainDF = pandas.DataFrame()
        trainDF['text'] = texts
        trainDF['label'] = labels
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], train_size = 0.8)
        encoder = preprocessing.LabelBinarizer()
        encoder.fit(trainDF['label'])
        TextClassificationTrain.save_classes('Data/Classes.txt', encoder.classes_)
        train_y = encoder.transform(train_y)
        valid_y = encoder.transform(valid_y)
        token = text.Tokenizer()
        token.fit_on_texts(trainDF['text'])
        train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=44)
        valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=44)
        embedding_dim, embedding_matrix = TextClassificationTrain.create_embedding_matrix(token.word_index)
        return train_seq_x, train_y, valid_seq_x, valid_y, token.word_index, embedding_matrix, embedding_dim, len(encoder.classes_)

def main():
    train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, embedding_dim, num_classes = TextClassificationTrain.prepare_data()
    TextClassificationTrain.run_training(BiGru, 'SavedModels/bigru.h5', train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, embedding_dim, num_classes, num_epochs, batch_size, learning_rate)
    TextClassificationTrain.run_training(Cnn, 'SavedModels/cnn.h5', train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, embedding_dim, num_classes, num_epochs, batch_size, learning_rate)

if __name__ == '__main__':
    main()