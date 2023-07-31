import tensorflow as tf
from dataset_manip import get_dataset_split_stringDB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import os


def obtain_data(confidence_score_pos, confidence_score_neg, go_id_dict_pth, go_embed_pth, shuffle, test_size):
    # get dataset
    train_embeddings, test_embeddings, train_labels, test_labels = get_dataset_split_stringDB(confidence_score_pos, confidence_score_neg, go_id_dict_pth, 
                                                                             go_embed_pth, shuffle, "", test_size)
    return train_embeddings, test_embeddings, train_labels, test_labels

def get_matrix(model, train_embeddings, train_labels):
    train_predictions = model.predict(train_embeddings)
    train_predicted_labels = np.argmax(train_predictions, axis=1)
    confusion_matrix_train = confusion_matrix(train_labels, train_predicted_labels)
    print(confusion_matrix)
    ConfusionMatrixDisplay(confusion_matrix_train).plot()
    plt.show()



if __name__ == '__main__':
    dim = 64
    confidence_score_pos = 901
    confidence_score_neg = 700
    model = tf.keras.models.load_model(r"results\RNN\binary\14-06-2023-simple_rnn_0_gpu_legit")
    os.chdir('recurrent_nn')
    go_id_dict_pth = "datasets\go_id_dict"
    go_embed_pth = f"datasets\go-terms-{dim}.emd"
    train_embeddings, test_embeddings, train_labels, test_labels = obtain_data(confidence_score_pos, confidence_score_neg, go_id_dict_pth, go_embed_pth, shuffle, 0)
    get_matrix(model, train_embeddings, train_labels)