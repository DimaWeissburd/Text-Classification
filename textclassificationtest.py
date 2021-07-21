import datetime
import csv
import pandas
from keras.preprocessing import text, sequence
from keras.models import load_model

class TextClassificationTest:
    def load_dataset(path):
        data = open(path, encoding='utf-8').read()
        texts, labels = [], []
        for line in data.split('\n'):
            content = line.split('\t')
            texts.append(content[0])
            labels.append(content[1])
        return texts, labels

    def make_predictions(model_path, test_data_path):
        texts, labels = TextClassificationTest.load_dataset(test_data_path)
        testDF = pandas.DataFrame()
        testDF['text'] = texts
        testDF['label'] = labels
        token = text.Tokenizer()
        token.fit_on_texts(testDF['text'])
        word_index = token.word_index
        test_seq = sequence.pad_sequences(token.texts_to_sequences(testDF['text']), maxlen=44)
        classifier = load_model(model_path)
        predictions = classifier.predict(test_seq)
        return texts, labels, predictions

    def load_classes():
        label_dict = {}
        fc = open('Data/Classes.txt', 'r', encoding='utf-8')
        for index, type in enumerate(fc):
            label_dict[index] = type.replace('\n', '')
        fc.close()
        return label_dict

    def compute_evaluations(index, labels_num, confusion_matrix):
        correct_answers = confusion_matrix[index][index]
        total_label_predictions = 0
        for i in range(labels_num):
            total_label_predictions += confusion_matrix[i][index]
        total_label_occurences = 0
        for i in range(labels_num):
            total_label_occurences += confusion_matrix[index][i]
        precision = 0
        if total_label_predictions != 0:
            precision = correct_answers/total_label_predictions
        recall = 0
        if total_label_occurences != 0:
            recall = correct_answers/total_label_occurences
        f1 = 0
        if (precision + recall) != 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1

    def evaluate_predictions(path, texts, labels, predictions):
        label_dict = TextClassificationTest.load_classes()
        confusion_matrix = [[0 for x in range(len(label_dict))] for y in range(len(label_dict))]
        correct_counter = 0
        key_list = list(label_dict.keys())
        val_list = list(label_dict.values())
        model_name = path.split('/')[1].split('.')[0]
        current_time = datetime.datetime.now()
        current_time_string = str(current_time.year) + '-' + str(current_time.month) + '-' + str(current_time.day) + '-' + str(current_time.hour) + '-' + str(current_time.minute) + '-' + str(current_time.second)
        output_log_path = 'Data/Log/' + model_name + '-' + current_time_string + '.csv'
        details_file = open(output_log_path, 'a', newline='', encoding='utf-8')
        details_csv_writer = csv.writer(details_file, dialect = 'excel', delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_file = open('Data/Log/Log.csv', 'a', newline='', encoding='utf-8')
        log_csv_writer = csv.writer(log_file, dialect = 'excel', delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_csv_writer.writerow([model_name, current_time_string])
        for i in range(len(predictions)):
            prediction = label_dict[predictions[i].argmax(axis=-1)]
            if (prediction == labels[i]):
                correct_counter+=1
            confusion_matrix[key_list[val_list.index(labels[i])]][key_list[val_list.index(prediction)]]+=1
        details_csv_writer.writerow([model_name, 'Correct predictions', str(correct_counter), 'Out of', str(len(predictions))])
        details_csv_writer.writerow(['Confusion matrix:'])
        for row in confusion_matrix:
            row_array = []
            for cell in row:
                row_array.append(cell)
            details_csv_writer.writerow(row_array)
        for index, label in enumerate(val_list):
            precision, recall, f1 = TextClassificationTest.compute_evaluations(index, len(val_list), confusion_matrix)
            details_csv_writer.writerow([label, 'Precision', str(precision)])
            details_csv_writer.writerow(['', 'Recall', str(recall)])
            details_csv_writer.writerow(['', 'F-measure', str(f1)])
            log_csv_writer.writerow(['', label, 'F-measure', str(f1)])
        details_csv_writer.writerow(['Several predictions:'])
        details_csv_writer.writerow(['Sentence', 'Prediction', 'Actual Label'])
        for i in range(len(predictions)):
            if i%10 == 0:
                details_csv_writer.writerow([texts[i], label_dict[predictions[i].argmax(axis=-1)], labels[i]])
        details_file.close()
        log_file.close()

    def predict_model(model_path, test_data_path):
        texts, labels, predictions = TextClassificationTest.make_predictions(model_path, test_data_path)
        TextClassificationTest.evaluate_predictions(model_path, texts, labels, predictions)

def main():
    TextClassificationTest.predict_model('SavedModels/bigru.h5', 'Data/Test.txt')
    TextClassificationTest.predict_model('SavedModels/cnn.h5', 'Data/Test.txt')

if __name__ == '__main__':
    main()
