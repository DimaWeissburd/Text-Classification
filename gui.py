import tkinter as tk
from tkinter import *
from textclassificationtrain import TextClassificationTrain
from textclassificationtest import TextClassificationTest
from Models.bigru import BiGru
from Models.cnn import Cnn

root = tk.Tk()
root.geometry("600x400")
root.title("Mobideo Text Classification")

#####################Train#####################

train_label = tk.Label(root, text="Train")
train_label.grid(row=0, column=0, sticky=NW)

model_name_label = tk.Label(root, text="Model name:")
model_name_label.grid(row=1,column=0, sticky=W)
model_name_string = StringVar(root)
model_name_string.set("BiGru")
model_name_dropdown = OptionMenu(root, model_name_string, "BiGru", "Cnn")
model_name_dropdown.grid(row=1,column=1, sticky=W)

model_save_path_label = tk.Label(root, text="Model saving path:")
model_save_path_label.grid(row=2,column=0, sticky=W)
model_save_path_textbox = tk.Entry(root, width=50)
model_save_path_textbox.insert(END, 'SavedModels/')
model_save_path_textbox.grid(row=2,column=1, sticky=W)

dataset_path_label = tk.Label(root, text="Dataset path:")
dataset_path_label.grid(row=3,column=0, sticky=W)
dataset_path_textbox = tk.Entry(root, width=50)
dataset_path_textbox.insert(END, 'Data/Train.txt')
dataset_path_textbox.grid(row=3,column=1, sticky=W)

num_epochs_label = tk.Label(root, text="Number of epochs:")
num_epochs_label.grid(row=4,column=0, sticky=W)
num_epochs_textbox = tk.Entry(root, width=50)
num_epochs_textbox.insert(END, '20')
num_epochs_textbox.grid(row=4,column=1, sticky=W)

batch_size_label = tk.Label(root, text="Batch size:")
batch_size_label.grid(row=5,column=0, sticky=W)
batch_size_textbox = tk.Entry(root, width=50)
batch_size_textbox.insert(END, '10')
batch_size_textbox.grid(row=5,column=1, sticky=W)

learning_rate_label = tk.Label(root, text="Learning rate:")
learning_rate_label.grid(row=6,column=0, sticky=W)
learning_rate_textbox = tk.Entry(root, width=50)
learning_rate_textbox.insert(END, '0.001')
learning_rate_textbox.grid(row=6,column=1, sticky=W)

def runTraining():
    train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, embedding_dim, num_classes = TextClassificationTrain.prepare_data()
    if (model_name_string.get() == "BiGru"):
        TextClassificationTrain.run_training(BiGru, model_save_path_textbox.get(), train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, embedding_dim, num_classes, int(num_epochs_textbox.get()), int(batch_size_textbox.get()), float(learning_rate_textbox.get()))
    if (model_name_string.get() == "Cnn"):
        TextClassificationTrain.run_training(Cnn, model_save_path_textbox.get(), train_seq_x, train_y, valid_seq_x, valid_y, word_index, embedding_matrix, embedding_dim, num_classes, int(num_epochs_textbox.get()), int(batch_size_textbox.get()), float(learning_rate_textbox.get()))

run_training_button = tk.Button(root, text='Run Training', command=runTraining)
run_training_button.grid(row=7,column=0, sticky=W)

#####################Test#####################

test_label = tk.Label(root, text="Test")
test_label.grid(row=8,column=0, sticky=W)

model_path_label = tk.Label(root, text="Model path:")
model_path_label.grid(row=9,column=0, sticky=W)
model_path_textbox = tk.Entry(root, width=50)
model_path_textbox.insert(END, 'SavedModels/')
model_path_textbox.grid(row=9,column=1, sticky=W)

test_dataset_path_label = tk.Label(root, text="Dataset path:")
test_dataset_path_label.grid(row=10,column=0, sticky=W)
test_dataset_path_textbox = tk.Entry(root, width=50)
test_dataset_path_textbox.insert(END, 'Data/Test.txt')
test_dataset_path_textbox.grid(row=10,column=1, sticky=W)

def runTest():
    TextClassificationTest.predict_model(model_path_textbox.get(), test_dataset_path_textbox.get())

run_training_button = tk.Button(root, text='Run Test', command=runTest)
run_training_button.grid(row=11,column=0, sticky=W)

##########################################

exit_button = tk.Button(root, text='Close', command=root.destroy)
exit_button.grid(row=12,column=0, sticky=W)

root.mainloop()