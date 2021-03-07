import os
import pandas, numpy
from sklearn import model_selection, preprocessing
from tqdm.keras import TqdmCallback
from keras.preprocessing import text, sequence
from keras import optimizers, backend
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from Models.creaternnlstm import CreateRnnLstm
from Models.creaternngru import CreateRnnGru
from Models.createbidirectionalrnn import CreateBidirectionalRnn
from Models.createcnn import CreateCnn
from utility import clean_string

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, num_epochs, save_file_name):
    # save model after each epoch if improvment was gained
    checkpoint = ModelCheckpoint(save_file_name, monitor='loss', verbose=1,
        save_weights_only = False, save_best_only=False, mode='auto', save_freq='epoch')
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label,
                    epochs=num_epochs,
                    validation_data=(feature_vector_valid, valid_y),
                    shuffle = True,
                    batch_size = 1,
                    verbose = 0,
                    callbacks=[checkpoint,
                               TqdmCallback(verbose=2)])

def run_training(model, name, model_file_name, train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, num_epochs, learning_rate, num_classes):
    if os.path.exists(model_file_name):
        classifier = load_model(model_file_name)
        backend.set_value(classifier.optimizer.lr, learning_rate)
    else:
        classifier = model.create(word_index, embedding_matrix, learning_rate, num_classes)
    train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, num_epochs, model_file_name)

def main():
    # load the dataset
    data = open('Data/Train.txt', encoding='cp1252').read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        content = line.split("\t")
        texts.append(content[0])
        labels.append(content[1])
    #for line in texts:
    #    line = clean_string(line)
    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels
    # split the dataset into training and validation datasets
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], train_size = 0.8)
    encoder = preprocessing.LabelBinarizer()
    encoder.fit(trainDF['label'])
    #print(encoder.classes_)
    num_classes = len(encoder.classes_)
    train_y = encoder.transform(train_y)
    valid_y = encoder.transform(valid_y)
    # create a tokenizer 
    token = text.Tokenizer()
    token.fit_on_texts(trainDF['text'])
    word_index = token.word_index
    # convert text to sequence of tokens and pad them to ensure equal length vectors 
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)
    # load the pre-trained word-embedding vectors
    embeddings_index = {}
    for i, line in enumerate(open('Data/glove.6B.300d.txt', 'r', encoding='utf-8', newline='\n', errors='ignore')):
        values = line.split()
        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
    # create token-embedding mapping
    embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    epochs = 10
    learning_rate = 0.001
    #lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.9)
    run_training(CreateRnnLstm, "RNN LSTM", 'SavedModels\\rnn_lstm_model.h5', train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, epochs, learning_rate, num_classes)
    run_training(CreateRnnGru, "RNN GRU", 'SavedModels\\rnn_gru_model.h5', train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, epochs, learning_rate, num_classes)
    run_training(CreateBidirectionalRnn, "Bidirectional RNN", 'SavedModels\\bidirectional_rnn_model.h5', train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, epochs, learning_rate, num_classes)
    run_training(CreateCnn, "CNN", 'SavedModels\\cnn_model.h5', train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, epochs, learning_rate, num_classes)

if __name__ == '__main__':
    main()