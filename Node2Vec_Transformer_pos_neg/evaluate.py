#import model
from transformer import *
from dataset_manip import *
from training_helper import *
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import pickle
from random import shuffle
from sklearn.metrics import roc_auc_score
from datetime import date
from tqdm import tqdm as prog_bar
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def helper_collate(batch):
    MAX_LEN_SEQ = get_max_len_seq(batch)
    return transformerGO_collate_fn(batch, MAX_LEN_SEQ, EMB_DIM, pytorch_pad = False)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    pred = []
    lab = []
    final = []
    
    correct_0 = []
    correct_1 = []
    false_0 = []
    false_1 = []
    
    model.eval()
    with torch.no_grad():
        for batch in prog_bar(iterator):
            proteins = []
            
            #padded pairs: tensor of shape N * 2(protein pair) * L(longest seq) * Emb dim
            padded_pairs = batch[0].to(device)
            labels = batch[1].to(device)
            mask = batch[2].to(device)

            for row in batch[:][3]:
                protA = row[0][0]
                protB = row[1][0]
                proteins.append((protA, protB))

            
            #split data into protA and protB
            gosetA_batch = padded_pairs[:,0]
            gosetB_batch = padded_pairs[:,1]
        
            predictions = model(gosetA_batch, gosetB_batch, mask[:,0], mask[:,1]).squeeze(1)
            
            final_predictions = np.round(torch.sigmoid(predictions))

            
            for i in range(len(labels)):
                if final_predictions[i] == 0:
                    if labels[i] == 0:
                        correct_0.append(proteins[i])
                    else:
                        false_0.append(proteins[i])
                else:
                    if labels[i] == 1:
                        correct_1.append(proteins[i])
                    else:
                        false_1.append(proteins[i])

            
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            pred = pred + list(predictions.cpu().data.numpy())
            lab = lab + list(labels.cpu().data.numpy())
            final = final + list(torch.round(torch.sigmoid(predictions)))

    return epoch_loss / len(iterator), epoch_acc / len(iterator), roc_auc_score(lab,pred), lab, pred, final, correct_0, correct_1, false_0, false_1


if __name__ == '__main__':    
    confidence_score_pos = 901
    confidence_score_neg = 700
    organism = [1, 2]
    EMB_DIM = 64
    
    # model
    model_path = "Cluster_results/27-05-2023-Neural_Network_Pos_Neg-total-model-3.pt"
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # paths
    os.chdir('Node2Vec_Transformer_pos_neg')    
    go_embed_pth = "datasets/go-terms-64.emd"
    go_id_dict_pth = "datasets/go_id_dict"
    go_name_space_dict_pth = "datasets/go_namespace_dict"
    
    # get dataset
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(confidence_score_pos, confidence_score_neg, go_id_dict_pth, 
                                                                             go_embed_pth, shuffle, "", organism=organism,  go_name_space_dict_pth=go_name_space_dict_pth, 
                                                                             ratio = [0.8, 0.2, 0], stringDB = False,)
    
    
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    
    c_val_grt = data.DataLoader(all_dataset, **params, shuffle = False)
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    loss, acc, roc, lab, pred, final, correct_0, correct_1, false_0, false_1 = evaluate(model, c_val_grt, criterion)
    results = {'correct_0': correct_0, 'correct_1': correct_1, 'false_0': false_0, 'false_1': false_1}
    with open('proteins_yeah.pkl', 'wb') as f:
        pickle.dump(results, f)
    # print(lab, final, sum(final), len(final))
    
    
