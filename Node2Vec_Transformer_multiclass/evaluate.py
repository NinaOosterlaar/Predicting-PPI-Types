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
from sklearn.metrics import roc_auc_score, accuracy_score
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
    correct_2 = []
    false_0_is_1 = []
    false_0_is_2 = []
    false_1_is_0 = []
    false_1_is_2 = []
    false_2_is_0 = []
    false_2_is_1 = []
    
    
    
    model.eval()
    with torch.no_grad():
        for batch in prog_bar(iterator):
            
            #padded pairs: tensor of shape N * 2(protein pair) * L(longest seq) * Emb dim
            padded_pairs = batch[0].to(device)
            labels = batch[1].to(device)
            mask = batch[2].to(device)
            
            proteins = []
            for row in batch[:][3]:
                protA = row[0][0]
                protB = row[1][0]
                proteins.append((protA, protB))
        
            #split data into protA and protB
            gosetA_batch = padded_pairs[:,0]
            gosetB_batch = padded_pairs[:,1]
        
            predictions = model(gosetA_batch, gosetB_batch, mask[:,0], mask[:,1]).squeeze(1)
            final_pred = F.softmax(predictions, dim = 1).type(torch.FloatTensor).to(device)
            predictions = predictions.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            # print(labels, predictions)
            loss = criterion(predictions, labels)
            idk = (torch.argmax(final_pred, dim = 1)).cpu().detach().numpy()
            acc = accuracy_score(labels.cpu().detach().numpy(), idk)
            
            for i in range(len(labels)):
                if idk[i] == 0:
                    if labels[i] == 0:
                        correct_0.append(proteins[i])
                    if labels[i] == 1:
                        false_0_is_1.append(proteins[i])
                    if labels[i] == 2:
                        false_0_is_2.append(proteins[i])
                elif idk[i] == 1:
                    if labels[i] == 0:
                        false_1_is_0.append(proteins[i])
                    if labels[i] == 1:
                        correct_1.append(proteins[i])
                    if labels[i] == 2:
                        false_1_is_2.append(proteins[i])
                else:
                    if labels[i] == 0:
                        false_2_is_0.append(proteins[i])
                    if labels[i] == 1:
                        false_2_is_1.append(proteins[i])
                    if labels[i] == 2:
                        correct_2.append(proteins[i])
                
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            pred = pred + list(final_pred.cpu().data.numpy())
            lab = lab + list(labels.cpu().data.numpy())
            final = final + np.argmax(final_pred.cpu().data.numpy(), axis = 1).tolist()
    results = {'correct_0': correct_0, 'correct_1': correct_1, 'correct_2': correct_2, 'false_0_is_1': false_0_is_1, 
               "false_0_is_2": false_0_is_2, "false_1_is_0": false_1_is_0, "false_1_is_2": false_1_is_2, "false_2_is_0": false_2_is_0, "false_2_is_1": false_2_is_1, "proteins": proteins}
    with open('proteins_yeah.pkl', 'wb') as f:
        pickle.dump(results, f)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), roc_auc_score(lab,pred, multi_class='ovr'), lab, pred, final


if __name__ == '__main__':    
    EMB_DIM = 64
    C_FOLD = 2
    N_EPOCHS = 5
    MODEL_SIZE = EMB_DIM
    NR_HEADS = 8
    NR_LAYERS = 3
    DROPOUT = 0.2
    SIZE_FF = 4 * MODEL_SIZE
    LR = 0.0001 
    
    interactions = [(1, 901), (2, 700), (9, 0)]
    
    model_path = r"results\Node2Vec\multiclass\13-06-2023-Neural_network_Multiclass-total-model-0.pt"
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # paths
    os.chdir('Node2Vec_Transformer_multiclass')    
    go_embed_pth = "datasets/go-terms-64.emd"
    go_id_dict_pth = "datasets/go_id_dict"
    go_name_space_dict_pth = "datasets/go_namespace_dict"
    
    # get dataset
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(interactions, go_id_dict_pth, 
                                                                             go_embed_pth, shuffle, "", go_name_space_dict_pth=go_name_space_dict_pth, 
                                                                             ratio = [0.8, 0, 0.2], stringDB = False, organism=[1, 2])
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    
    c_val_grt = data.DataLoader(all_dataset, **params, shuffle = False)
    criterion = nn.CrossEntropyLoss().to(device)
    
    evaluate(model, c_val_grt, criterion)