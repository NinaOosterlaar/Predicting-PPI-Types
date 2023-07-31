import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
from tqdm import tqdm as prog_bar
import torch.utils.data as data
from random import shuffle
import sys
import pandas as pd

"""Change path to folder you want to use"""

sys.path.append(r"C:\Users\\ninao\OneDrive - Delft University of Technology\Nanobiology\Bachelor End Project\Bep\Node2Vec_Transformer_pos_neg")
from evaluate import evaluate, helper_collate
from dataset_manip import get_dataset_split_stringDB2, get_max_len_seq
from training_helper import transformerGO_collate_fn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_GO_explanation(go_terms, obo_csv_path = "..\\term-encoding-module\\go-basic.obo.csv"):
    go_terms_csv = pd.read_csv(obo_csv_path)
    go_terms = [ "['" + go + "']" for go in go_terms ]
    go_terms_csv = go_terms_csv.query('id == @go_terms')
    
    #do the query one by one to keep the order (using list query does not maintain it)
    final_query = go_terms_csv.query('id == @go_terms[0]')
    for i in range(1, len(go_terms)):
        go = go_terms[i]
        query = go_terms_csv.query('id == @go')
        final_query = pd.concat([final_query, query], ignore_index=True)
    return final_query

def trim_GO_expl(expls):
    return [expl.strip("'[").strip("]'") for expl in expls ]

def process_to_capital(onthologies):
    cap = {"['biological_process']":"BP", "['cellular_component']":"CC", "['molecular_function']": "MF"}
    return [cap[ontho] for ontho in onthologies] 

def idstogos(ids, go_id_dict):
    key_list = list(go_id_dict.keys())
    val_list = list(go_id_dict.values())  
    return [key_list[val_list.index(go_id)] for go_id in ids]

def gostoids(gos, go_id_dict):
    return [go_id_dict[go] for go in gos]



def helper_collate(batch):
    MAX_LEN_SEQ = get_max_len_seq(batch)
    return transformerGO_collate_fn(batch, MAX_LEN_SEQ, EMB_DIM, pytorch_pad = False)




def get_attn(layers, layer = 0, head = 0, avg = "HEAD", src_attn = False):     
    """ Function that extracts the weights from the attention mechanism
    Args:
        layers (model layers): layers from the encoder or decoder
        layer (int): index of the layer
        head (int): index of the head
        avg (string): type of averaging of the attention: 
                        "NONE", average over heads "HEAD", average over all layers "Layer", average over all layers and heads "ALL"
        src_attn (bool): the extraction of source attention from the layers or the self attention

    Returns:
    numpy: numpy containing the averaged attention
    """
        
    if src_attn:
        attn = layers[layer].src_attn.attn
    else:
        attn = layers[layer].self_attn.attn
    if avg == "NONE":
        return attn[0, head].data
    if avg == "HEAD":
        return torch.mean(attn[0], dim = 0)
    
    
    attn = torch.zeros(layers[0].self_attn.attn.shape).to(device)
    for l in layers:
        if src_attn:
            attn = torch.cat((attn, l.src_attn.attn))
        else:
            attn = torch.cat((attn, l.self_attn.attn))
    
    attn = torch.mean(attn, dim = 0)
    if avg == "LAYER":
        return attn[head]
    elif avg == "ALL":
        return torch.mean(attn, dim = 0)
    
    
def add_to_heatmap(heatm, go_id_dict, protA, protB, attn):
    
    """ Function that adds attention weights to a heatmap
    Args:
        heatm (numpy): matrix to add the attention and freq to
        go_id_dict (dict): dictionary of mapping 'GO-name -> 1'
        protA (list): list of GO terms annotated to protA
        protB (list): list of GO terms annotated to protB
    """
        
    for i in range(len(protA)):
        go1 = go_id_dict[protA[i]]
        for j in range(len(protB)):
            go2 = go_id_dict[protB[j]]

            #attn 
            heatm[go1,go2][0] +=  attn[i][j].cpu()
            
            # frequency
            heatm[go1,go2][1] +=  1
    
    
def get_heamaps(go_id_dict, model, train_grt):
    
    """ Function that generates the M attention matrix (both self and source) containing all the attention
        values between the GO terms in the given dataset
    Args:
        go_id_dict (dict): Mapping between go name and id e.g. GO1 -> 1
        model (TransformerGO): trained model which is used to analyse the weigths
        train_grt (Dataset_stringDB): the dataset used for training the model

    Returns:
    numpy: Attention matrix for the source and self attention 
    """
    
    l = len(go_id_dict)
    # 2 = (attn_sum, number_of_attn_added_to_sum)
    go_heatm_self = np.zeros(l * l * 2).reshape(l,l,2)
    go_heatm_source = np.zeros(l * l * 2).reshape(l,l,2)

    model.eval()
    with torch.no_grad():
            added = 0
            for batch in prog_bar(train_grt):
                #padded pairs: tensor of shape N * 2(protein pair) * L(longest seq) * Emb dim
                padded_pairs = batch[0].to(device)
                labels = batch[1].to(device)
                mask = batch[2].to(device)
                batch_ids = batch[3]
            
                #swap seqLen with batch to fit the transformer
                gosetA_batch = padded_pairs[:,0]
                gosetB_batch = padded_pairs[:,1] 
            
                #only for 1 protein pair at a time
                for i in range(0, batch[0].shape[0]):
                    
                    protA = batch_ids[i][0][1]
                    protB = batch_ids[i][1][1]
                    
                        
                    added += 1
                    pred = model(gosetA_batch[i].unsqueeze(0), gosetB_batch[i].unsqueeze(0), \
                                            mask[i,0].unsqueeze(0), mask[i,1].unsqueeze(0)).squeeze(1)

                    attn = get_attn(model.encoder.layers, avg = "ALL")
                    add_to_heatmap(go_heatm_self, go_id_dict, protA, protA, attn)

                    attn = get_attn(model.decoder.layers, avg = "ALL")
                    add_to_heatmap(go_heatm_self, go_id_dict, protB, protB, attn)
                    
                    attn = get_attn(model.decoder.layers, avg = "ALL", src_attn = True)
                    add_to_heatmap(go_heatm_source, go_id_dict, protB, protA, attn) 
                     
            print("Number of interactions added:", added)
    return go_heatm_self, go_heatm_source


def get_top_go(go_heatm, number_of_go,  dim = 1):
    #retrieving the freq from the (l,l,2) matrix
    go_freq = go_heatm[:,:,1]
    #retrieving the attention sum
    go_heatmap = go_heatm[:,:,0]
    
    go_heatm_sc = go_heatmap/np.where(go_freq==0, 1e9, go_freq)
    
    sumed_attn_dic, indxs = zip(*sorted(zip(np.sum(go_heatm_sc, axis = 1), np.arange(len(go_heatm_sc))), key=lambda x: x[0]))
    
    print(len(indxs))
    
    return indxs[-number_of_go:]


def load_filtered_go_id_dict(name):
    with open(name, 'rb') as fp:
        go_id_dict = pickle.load(fp)
    return go_id_dict 

    
if __name__ == '__main__':
    """Change path to the model you want to analyse"""
    model_path = 'results/Node2Vec/binary/20-05-2023-Neural_Network_Pos_Neg-total-model-0.pt'
    go_embed_pth = "Node2Vec_Transformer_pos_neg/datasets/go-terms-64.emd"
    go_id_dict_pth = "Node2Vec_Transformer_pos_neg/datasets/go_id_dict"
    go_name_space_dict_pth = "Node2Vec_Transformer_pos_neg/datasets/go_namespace_dict"
    
    go_dic = load_filtered_go_id_dict(r'results\analysis\go_id_dict_filtered_yeast')
    
    model = torch.load(model_path, map_location=torch.device('cpu'))
    EMB_DIM = 64
    
    confidence_score_pos = 980
    confidence_score_neg = 950
    
    organism = [1]
    
    # get dataset
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB2(confidence_score_pos, confidence_score_neg, go_id_dict_pth, 
                                                                             go_embed_pth, shuffle, go_dic, "", go_name_space_dict_pth=go_name_space_dict_pth, 
                                                                             ratio = [0.8, 0, 0.2], new_thingy = 'results/analysis/new_dict.pkl', stringDB = False, organism = organism)
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    train_grt = data.DataLoader(all_dataset, **params, shuffle = False)
    
    go_heatm_self, go_heatm_source = get_heamaps(go_dic, model, train_grt)
    ids = get_top_go(go_heatm_source, 30)
    
    print(sorted(ids))
    labels = idstogos(ids, go_dic)
    expls = get_GO_explanation(labels, obo_csv_path = "results/analysis/go-basic.obo.csv")
    # print(labels)
    # print(expls)
    