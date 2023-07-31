from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from dataset_manip import get_dataset_split_stringDB
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from random import shuffle
import datetime
import pickle


def create_rnn_model(emb_dim, summary=False):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(None, emb_dim)))
    model.add(SimpleRNN(units=64, activation='relu', return_sequences=True))
    model.add(SimpleRNN(units=64, activation='relu', return_sequences=True))
    model.add(SimpleRNN(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    if summary:
        model.summary()
    
    return model
    

def obtain_data(confidence_score_pos, confidence_score_neg, go_id_dict_pth, go_embed_pth, shuffle, test_size):
    # get dataset
    train_embeddings, test_embeddings, train_labels, test_labels = get_dataset_split_stringDB(confidence_score_pos, confidence_score_neg, go_id_dict_pth, 
                                                                             go_embed_pth, shuffle, "", test_size)
    return train_embeddings, test_embeddings, train_labels, test_labels


def cross_train(c_fold, epoch, dimension, train_embeddings, test_embeddings, train_labels, test_labels):
    sz = len(train_embeddings)
    fold_size = int(sz/c_fold)
    l = 0
    r = fold_size
    
    test_set_padded = pad_sequences(test_embeddings, padding='post')
    c_test = tf.convert_to_tensor(test_set_padded)

    train_accs = []
    train_rocs = [] 
    val_accs = []
    val_rocs = []
    final_test = []
    final_test_roc = []
    
    print(train_embeddings.shape)
    for i in range(c_fold):
        print("Fold nr: ", i)
        
        if i == 0:
            train_set = train_embeddings[r:]
            train_labels_final = train_labels[r:]
            valid_set = train_embeddings[l:r]
            valid_labels = train_labels[l:r]
        else: 
            train_set = np.concatenate((train_embeddings[:l], train_embeddings[r:]), axis=0)
            train_labels_final = np.concatenate((train_labels[:l], train_labels[r:]), axis=0)
            valid_set = train_embeddings[l:r]
            valid_labels = train_labels[l:r]
            
        # Pad sequences to a fixed length
        train_set_padded = pad_sequences(train_set, padding='post')
        valid_set_padded = pad_sequences(valid_set, padding='post')
        
        # Convert to TensorFlow tensors
        train_set = tf.convert_to_tensor(train_set_padded)
        train_labels_final = tf.convert_to_tensor(train_labels_final)
        valid_set = tf.convert_to_tensor(valid_set_padded)
        valid_labels = tf.convert_to_tensor(valid_labels)
            
        # Train the model
        model = create_rnn_model(dimension)
        model.fit(train_set, train_labels_final, epochs=epoch, batch_size=32, validation_data=(valid_set, valid_labels))
        
        print("finished training")
        
        # Validate on the validation set and test set
        train_loss, train_acc = model.evaluate(train_set, train_labels_final)
        train_roc = roc_auc_score(train_labels_final, model.predict(train_set))
        train_accs.append(train_acc)
        train_rocs.append(train_roc)
        val_loss, val_acc = model.evaluate(valid_set, valid_labels)
        val_roc = roc_auc_score(valid_labels, model.predict(valid_set))
        val_accs.append(val_acc)
        val_rocs.append(val_roc)    
        test_loss, test_acc = model.evaluate(c_test, test_labels)
        test_roc = roc_auc_score(test_labels, model.predict(c_test))
        final_test.append(test_acc)
        final_test_roc.append(test_roc)

        # Update the fold boundaries
        l += fold_size
        r += fold_size
        
        current_day = datetime.date.today().strftime("%d-%m-%Y")
        
        model.save(f"binary_models\{current_day}-simple_rnn_{i}")
    
    result = {'train_acc': train_accs, 'train_roc': train_rocs, 'val_acc': val_accs, 'val_roc': val_rocs, 'test_acc': final_test, 'test_roc': final_test_roc}
    
    with open(f"binary_models\{current_day}-simple_rnn_result", "wb") as f:
        pickle.dump(result, f)
        
        
        
    
def main():
    dim = 64
    c_fold = 5
    epoch = 10
    confidence_score_pos = 900
    confidence_score_neg = 700
    go_id_dict_pth = "datasets\go_id_dict"
    go_embed_pth = f"datasets\go-terms-{dim}.emd"
    test_size = 0.2
    train_embeddings, test_embeddings, train_labels, test_labels = obtain_data(confidence_score_pos, confidence_score_neg, go_id_dict_pth, go_embed_pth, shuffle, test_size)
    cross_train(c_fold, epoch, dim, train_embeddings, test_embeddings, train_labels, test_labels)

# # Train the model
# model.fit(X, y, epochs=10, batch_size=32)

# # Make predictions
# predictions = model.predict(X)


if __name__ == '__main__':
    os.chdir('recurrent_nn')
    main()
    with open("binary_models/31-05-2023-simple_rnn_result", "rb") as f:
        result = pickle.load(f)
        print(result)

# tf.keras.layers.SimpleRNN(
#     units,
#     activation='tanh',
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     recurrent_initializer='orthogonal',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     recurrent_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     recurrent_constraint=None,
#     bias_constraint=None,
#     dropout=0.0,
#     recurrent_dropout=0.0,
#     return_sequences=False,
#     return_state=False,
#     go_backwards=False,
#     stateful=False,
#     unroll=False,
#     **kwargs
# )

