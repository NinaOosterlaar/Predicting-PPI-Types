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
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import date
from tqdm import tqdm as prog_bar
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from joblib import Parallel, delayed
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
    for batch in iterator:
        optimizer.zero_grad()
        
        #padded pairs: tensor of shape N * 2(protein pair) * L(longest seq) * Emb dim
        padded_pairs = batch[0].to(device)
        labels = batch[1].to(device)
        mask = batch[2].to(device)
        
        #split data into protA and protB
        gosetA_batch = padded_pairs[:,0]
        gosetB_batch = padded_pairs[:,1]
        
        predictions = model(gosetA_batch, gosetB_batch, mask[:,0], mask[:,1]).squeeze(1)
        labels = labels.type(torch.LongTensor).to(device)
        predictions = predictions.type(torch.FloatTensor).to(device)
        loss = criterion(predictions, labels)
        idk = (torch.argmax(predictions, dim = 1)).cpu().detach().numpy()
        acc = accuracy_score(labels.cpu().detach().numpy(), idk)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        pred = pred + list(predictions.cpu().data.numpy())
        lab = lab + list(labels.cpu().data.numpy())
 
    return epoch_loss / len(iterator), epoch_acc / len(iterator), roc_auc_score(lab,pred, multi_class='ovr')


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
            predictions = predictions.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            predictions = predictions.type(torch.FloatTensor).to(device)
            loss = criterion(predictions, labels)
            idk = (torch.argmax(predictions, dim = 1)).cpu().detach().numpy()
            acc = accuracy_score(labels.cpu().detach().numpy(), idk)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            pred = pred + list(predictions.cpu().data.numpy())
            lab = lab + list(labels.cpu().data.numpy())

    return epoch_loss / len(iterator), epoch_acc / len(iterator), roc_auc_score(lab,pred, multi_class='ovr'), lab, pred



def cross_train(NUM_CLASS, N_EPOCHS, MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, all_dataset, test_set, params, name, count, total):
    sz = len(all_dataset)
    fold_size = int(sz/total)
    l = count * fold_size
    r = fold_size + count * fold_size
    indexes = np.arange(sz)

    val_accs = []
    val_rocs = []
    final_test = []
    total_times = []
    
    test_set_final = data.DataLoader(test_set, **params, shuffle = False)
    name = name +  f"-model_size-{MODEL_SIZE},heads-{NR_HEADS},nr_layers-{NR_LAYERS},size_ff-{SIZE_FF},dropout-{DROPOUT},epoch-{N_EPOCHS},LR-{LR}"
    print(f"{name} started training")
    
    
    total_time = 0

    model = TransformerGO_Scratch(MODEL_SIZE, NR_HEADS, NR_LAYERS, SIZE_FF, NUM_CLASS, DROPOUT).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=LR) 
    criterion = nn.CrossEntropyLoss().to(device)

    val_subset = data.Subset(all_dataset, indexes[l:r])
    c_val_grt = data.DataLoader(val_subset, **params, shuffle = False)

    train_subset = data.Subset(all_dataset, np.concatenate([indexes[0:l], indexes[r:sz]]))
    c_train_grt = data.DataLoader(train_subset, **params, shuffle = True)


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
        
        best_valid_roc = max(best_valid_roc, roc_val)
        best_valid_acc = max(best_valid_acc, valid_acc)
        
        total_time += end_time - start_time

    val_rocs.append(best_valid_roc)
    val_accs.append(best_valid_acc)

    torch.save(model.state_dict(), f"{name}.pt")
    torch.save(model, f"{name}-total-model.pt")

    loss, accuracy, roc, lab, pred = evaluate(model, test_set_final, criterion)
    final_test.append((loss, accuracy, roc))

    model_params = {"MODEL_SIZE": MODEL_SIZE, "NR_HEADS": NR_HEADS, "NR_LAYERS": NR_LAYERS, "SIZE_FF": SIZE_FF, "DROPOUT": DROPOUT, 
                    "LR": LR, "N_EPOCHS": N_EPOCHS, "train_loss": train_loss, "train_acc": train_acc, "roc_train": roc_train,
                    "valid_loss": valid_loss, "valid_acc": valid_acc, "roc_val": roc_val, "test_loss": loss, "test_accuracy": accuracy, "test_roc": roc}
    
    
    torch.save(model.state_dict(), f"{name}.pt")
    torch.save(model, f"{name}-total-model.pt")
    with open(name + ".pkl", "wb") as f:
        pickle.dump(model_params, f)
        
    return name, train_loss, train_acc, roc_train, valid_loss, valid_acc, roc_val, loss, accuracy, roc
        

def train_epoch(MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, interactions, go_embed_pth, shuffle, name):
    N_EPOCHS = [1, 1, 1]
    NUM_CLASS = len(interactions)
    
    go_embed_pth = "datasets\sentence_embedding.json"
    
    # get dataset
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(interactions, go_embed_pth, shuffle, ratio = [0.8, 0, 0.2])
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    
    total = len(N_EPOCHS)
    
    Parallel(n_jobs=-1)(
        delayed(cross_train)(
            NUM_CLASS, N_EPOCH, MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, train_set, test_set, params, name, count, total
        ) for count, N_EPOCH in enumerate(N_EPOCHS)
    )
    

def train_heads(MODEL_SIZE, N_EPOCH, NR_LAYERS, DROPOUT, SIZE_FF, LR, interactions, go_embed_pth, shuffle, name):
    changeable= [4, 8, 16]
    NUM_CLASS = len(interactions)

    
    go_embed_pth = "datasets\sentence_embedding.json"
    
    # get dataset
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(interactions, go_embed_pth, shuffle, ratio = [0.8, 0, 0.2])
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    
    total = len(changeable)
    
    Parallel(n_jobs=-1)(
        delayed(cross_train)(
            NUM_CLASS, N_EPOCH, MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, train_set, test_set, params, name, count, total
        ) for count, NR_HEADS in enumerate(changeable)
    )
    
    
def train_layers(MODEL_SIZE, N_EPOCH, NR_HEADS, DROPOUT, SIZE_FF, LR, interactions, go_embed_pth, shuffle, name):
    changeable= [2, 3, 4, 5, 6, 7, 8]
    NUM_CLASS = len(interactions)

    
    go_embed_pth = "datasets\sentence_embedding.json"
    
    # get dataset
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(interactions, go_embed_pth, shuffle, ratio = [0.8, 0, 0.2])
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    
    total = len(changeable)
    
    Parallel(n_jobs=-1)(
        delayed(cross_train)(
            NUM_CLASS, N_EPOCH, MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, train_set, test_set, params, name, count, total
        ) for count, NR_LAYERS in enumerate(changeable)
    )
    

def train_dropout(MODEL_SIZE, N_EPOCH, NR_HEADS, NR_LAYERS, SIZE_FF, LR, interactions, go_embed_pth, shuffle, name):
    changeable= [0.1, 0.2, 0.3, 0.4, 0.5]
    NUM_CLASS = len(interactions)

    
    go_embed_pth = "datasets\sentence_embedding.json"
    
    # get dataset
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(interactions, go_embed_pth, shuffle, ratio = [0.8, 0, 0.2])
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    total = len(changeable)
    
    Parallel(n_jobs=-1)(
        delayed(cross_train)(
            NUM_CLASS, N_EPOCH, MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, train_set, test_set, params, name, count, total
        ) for count, DROPOUT in enumerate(changeable)
    )
    

def train_size_ff(MODEL_SIZE, N_EPOCH, NR_HEADS, NR_LAYERS, DROPOUT, LR, interactions, go_embed_pth, shuffle, name):
    changeable= [1* MODEL_SIZE, 2* MODEL_SIZE, 4 * MODEL_SIZE, 6 * MODEL_SIZE, 8 * MODEL_SIZE]
    NUM_CLASS = len(interactions)


    go_embed_pth = "datasets\sentence_embedding.json"
    
    # get dataset
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(interactions, go_embed_pth, shuffle, ratio = [0.8, 0, 0.2])
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    
    total = len(changeable)
    
    Parallel(n_jobs=-1)(
        delayed(cross_train)(
            NUM_CLASS, N_EPOCH, MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, train_set, test_set, params, name, count, total
        ) for count, SIZE_FF in enumerate(changeable)
    )
    

def train_LR(MODEL_SIZE, N_EPOCH, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, interactions, go_embed_pth, shuffle, name):
    changeable= [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    NUM_CLASS = len(interactions)

    
    go_embed_pth = "datasets\sentence_embedding.json"
    
    # get dataset
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(interactions, go_embed_pth, shuffle, ratio = [0.8, 0, 0.2])
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    total = len(changeable)
    
    Parallel(n_jobs=-1)(
        delayed(cross_train)(
            NUM_CLASS, N_EPOCH, MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, train_set, test_set, params, name, count, total
        ) for count, LR in enumerate(changeable)
    )
    
    
    
if __name__ == '__main__':
    # Hyperparamers
    EMB_DIM = 700
    C_FOLD = 2
    N_EPOCHS = 1
    MODEL_SIZE = EMB_DIM
    NR_HEADS = 14
    NR_LAYERS = 3
    DROPOUT = 0.2
    SIZE_FF = 4 * MODEL_SIZE
    LR = 0.0001 
    
    interactions = [(1, 990), (2, 990), (9, 0)]
    params = {'batch_size': 32, 'collate_fn': helper_collate}

    # paths
    os.chdir('Sent2Vec_Transformer_multiclass')    
    go_embed_pth = "datasets\sentence_embedding.json"
    

    # Start runs
    name = "Models/" + date.today().strftime("%d-%m-%Y")
    
    train_epoch(MODEL_SIZE, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, LR, interactions, go_embed_pth, shuffle, name)
    # train_heads(MODEL_SIZE, N_EPOCHS, NR_LAYERS, DROPOUT, SIZE_FF, LR, interactions, go_embed_pth, shuffle, name)
    # train_layers(MODEL_SIZE, N_EPOCHS, NR_HEADS, DROPOUT, SIZE_FF, LR, interactions, go_embed_pth, shuffle, name)
    # train_dropout(MODEL_SIZE, N_EPOCHS, NR_HEADS, NR_LAYERS, SIZE_FF, LR, interactions, go_embed_pth, shuffle, name)
    # train_size_ff(MODEL_SIZE, N_EPOCHS, NR_HEADS, NR_LAYERS, DROPOUT, LR, interactions, go_embed_pth, shuffle, name)
    # train_LR(MODEL_SIZE, N_EPOCHS, NR_HEADS, NR_LAYERS, DROPOUT, SIZE_FF, interactions, go_embed_pth, shuffle, name)