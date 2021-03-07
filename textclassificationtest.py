import pandas
from keras.preprocessing import text, sequence
from keras.models import load_model
from utility import clean_string

def predict_model(path, model_name):
    data = open('Data/Test.txt', encoding='cp1252').read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        content = line.split("\t")
        texts.append(content[0])
        labels.append(content[1])
    #for line in texts:
    #    line = clean_string(line)
    testDF = pandas.DataFrame()
    testDF['text'] = texts
    testDF['label'] = labels
    token = text.Tokenizer()
    token.fit_on_texts(testDF['text'])
    word_index = token.word_index
    test_seq = sequence.pad_sequences(token.texts_to_sequences(testDF['text']), maxlen=70)
    classifier = load_model(path)
    predictions = classifier.predict(test_seq)
    label_dict = {
        0 : "Tag1",
        1 : "Tag2",
        2 : "Tag3",
        3 : "Tag4"
    }
    correct_counter = 0
    f = open("Data/Output.txt", "a")
    f.write("****Start Prediction using " + model_name + " ****\n")
    for i in range(len(predictions)):
        #f.write("Text: " + texts[i] + " | Prediction: " + label_dict[predictions[i].argmax(axis=-1)] + " | Actual Label: " + labels[i] + "\n")
        if (label_dict[predictions[i].argmax(axis=-1)] == labels[i]):
            correct_counter+=1
    f.write(model_name + " model predicted correctly " + str(correct_counter) + " out of " + str(len(predictions)) + " sentences.\n")
    f.write("****End Prediction****\n")
    f.close()

def main():
    predict_model('SavedModels\\bidirectional_rnn_model.h5', "Bidirectional RNN")
    predict_model('SavedModels\\cnn_model.h5', "CNN")
    predict_model('SavedModels\\rnn_gru_model.h5', "RNN GRU")
    predict_model('SavedModels\\rnn_lstm_model.h5', "RNN LSTM")

if __name__ == '__main__':
    main()
