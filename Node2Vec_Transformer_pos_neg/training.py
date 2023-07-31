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
import time
import pickle
from random import shuffle
from sklearn.metrics import roc_auc_score
from datetime import date
from tqdm import tqdm as prog_bar
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Device available: ", device, " ", torch.cuda.get_device_name(0))

#for np array from nested sequences
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def helper_collate(batch):
    MAX_LEN_SEQ = get_max_len_seq(batch)
    return transformerGO_collate_fn(batch, MAX_LEN_SEQ, EMB_DIM, pytorch_pad = False)


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    pred = []
    lab = []
    for batch in prog_bar(iterator):
        optimizer.zero_grad()
        
        #padded pairs: tensor of shape N * 2(protein pair) * L(longest seq) * Emb dim
        padded_pairs = batch[0].to(device)
        labels = batch[1].to(device)
        mask = batch[2].to(device)
        
        #split data into protA and protB
        gosetA_batch = padded_pairs[:,0]
        gosetB_batch = padded_pairs[:,1]
        
        predictions = model(gosetA_batch, gosetB_batch, mask[:,0], mask[:,1]).squeeze(1)
        print(predictions)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        pred = pred + list(predictions.cpu().data.numpy())
        lab = lab + list(labels.cpu().data.numpy())
 
    return epoch_loss / len(iterator), epoch_acc / len(iterator), roc_auc_score(lab,pred)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    pred = []
    lab = []
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            
            #padded pairs: tensor of shape N * 2(protein pair) * L(longest seq) * Emb dim
            padded_pairs = batch[0].to(device)
            labels = batch[1].to(device)
            mask = batch[2].to(device)
        
            #split data into protA and protB
            gosetA_batch = padded_pairs[:,0]
            gosetB_batch = padded_pairs[:,1]
        
            predictions = model(gosetA_batch, gosetB_batch, mask[:,0], mask[:,1]).squeeze(1)
            print(predictions)
            print(labels)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            pred = pred + list(predictions.cpu().data.numpy())
            lab = lab + list(labels.cpu().data.numpy())

    return epoch_loss / len(iterator), epoch_acc / len(iterator), roc_auc_score(lab,pred), lab, pred


def cross_train(C_FOLD, N_EPOCHS, MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, all_dataset, test_set, params, name):
    sz = len(all_dataset)
    fold_size = int(sz/C_FOLD)
    l = 0
    r = fold_size
    indexes = np.arange(sz)

    val_accs = []
    val_rocs = []
    final_test = []
    
    test_set_final = data.DataLoader(test_set, **params, shuffle = False)
    for i in range(0, C_FOLD):
        print("Fold nr: ", i, end='\r')
        
        model = TransformerGO_Scratch(MODEL_SIZE, NR_HEADS, NR_LAYERS, SIZE_FF, DROPOUT).to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        optimizer = optim.Adam(model.parameters(), lr=LR) 
        criterion = nn.BCEWithLogitsLoss().to(device)
        writer = SummaryWriter(flush_secs=14)
        
        val_subset = data.Subset(all_dataset, indexes[l:r])
        c_val_grt = data.DataLoader(val_subset, **params, shuffle = False)
        
        train_subset = data.Subset(all_dataset, np.concatenate([indexes[0:l], indexes[r:sz]]))
        c_train_grt = data.DataLoader(train_subset, **params, shuffle = True)
        
        l += fold_size
        r += fold_size

        best_valid_roc = float('-inf')
        best_valid_acc = float('-inf')
        for epoch in range(N_EPOCHS):

            start_time = time.time()
            train_loss, train_acc, roc_train = train(model, c_train_grt, optimizer, criterion)
            valid_loss, valid_acc, roc_val, _, _ = evaluate(model, c_val_grt, criterion)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
            print_status(epoch, epoch_mins, epoch_secs, train_loss,\
                    train_acc, valid_loss, valid_acc, roc_train, roc_val, optimizer)
            write_scalars_tensorboard(writer, train_loss, valid_loss, train_acc, valid_acc, epoch)
            
            best_valid_roc = max(best_valid_roc, roc_val)
            best_valid_acc = max(best_valid_acc, valid_acc)

        val_rocs.append(best_valid_roc)
        val_accs.append(best_valid_acc)
        
        torch.save(model.state_dict(), f"{name}-{i}.pt")
        torch.save(model, f"{name}-total-model-{i}.pt")
        
        loss, accuracy, roc, lab, pred = evaluate(model, test_set_final, criterion)
        final_test.append((loss, accuracy, roc))
        
        
    model_params = {"MODEL_SIZE": MODEL_SIZE, "NR_HEADS": NR_HEADS, "NR_LAYERS": NR_LAYERS, "SIZE_FF": SIZE_FF, "DROPOUT": DROPOUT, 
                    "LR": LR, "C_FOLD": C_FOLD, "N_EPOCHS": N_EPOCHS, "val_rocs": val_rocs, "val_accs": val_accs, "final_test": final_test}

    with open(name + "-model_params.pkl", "wb") as f:
        pickle.dump(model_params, f)
        
        
        
    


if __name__ == '__main__':
        # Hyperparameters
    EMB_DIM = 64
    C_FOLD = 2
    N_EPOCHS = 2
    MODEL_SIZE = EMB_DIM
    NR_HEADS = 8
    NR_LAYERS = 3
    DROPOUT = 0.2
    SIZE_FF = 4 * MODEL_SIZE
    LR = 0.0001 
    
    confidence_score_pos = 985
    confidence_score_neg = 950

    # paths
    os.chdir('Node2Vec_Transformer_pos_neg')    
    go_embed_pth = "datasets/go-terms-64.emd"
    go_id_dict_pth = "datasets/go_id_dict"
    go_name_space_dict_pth = "datasets/go_namespace_dict"
    
    # get dataset
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(confidence_score_pos, confidence_score_neg, go_id_dict_pth, 
                                                                             go_embed_pth, shuffle, "", go_name_space_dict_pth=go_name_space_dict_pth, 
                                                                             ratio = [0.8, 0, 0.2], stringDB = False)
    params = {'batch_size': 32, 'collate_fn': helper_collate}

    # Start run
    print("Enter title: ")
    title = input()
    name = "Models/" + date.today().strftime("%d-%m-%Y") + "-" + title
    cross_train(C_FOLD, N_EPOCHS, MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, train_set, test_set, params, name)
    
    
   
    
