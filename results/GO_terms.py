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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sys.path.append(r"C:\Users\\ninao\OneDrive - Delft University of Technology\Nanobiology\Bachelor End Project\Bep\Sent2Vec_Transformer_pos_neg")
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


def get_top_sorted_attn(go_heatm_sc, number_of_go, axis):
    
    """ Function that selects the GO indexes in the heatmap that have the highest attention
    Args:
        go_heatm_sc (numpy): heatmap of the GO terms
        number_of_go (int): how many GO terms to select from the highest attention values
        axis (int): attention is summed over columns or rows 

    Returns:
    list: GO indexes values coresponding to the rows and columns in the heatmap
    """
   
    #summed attention with position in matrix
    sumed_attn_dic, indxs = zip(*sorted(zip(np.sum(go_heatm_sc, axis = axis), np.arange(len(go_heatm_sc))), key=lambda x: x[0]))
    
    # print(indxs)
    
    #return the top go terms with high attention 
    return indxs[-number_of_go:]


def get_filtered_indexes(go_heatm_sc, number_of_go, top = True, attn = "FOCUSED"):
    
    """ Function that selects the GO indexes in the heatmap that have the highest attention
    Args:
        go_heatm_sc (numpy): heatmap of the GO terms
        number_of_go (int): how many GO terms to select from the highest attention values
        top (bool): choose the top GO terms with high attention or those with low attention
        attn (string): "FOCUSED" means calulating the attention over columns, while "CONTRIBUTED" over rows
 
    Returns:
    list: GO indexes values coresponding the rows and columns in the heatmap
    """
    
    assert attn == "FOCUSED" or attn == "CONTRIBUTED", \
          "Invalid attn input, it should be FOCUSED or CONTRIBUTED"

    #return the top go terms
    if attn == "FOCUSED":
        return get_top_sorted_attn(go_heatm_sc, number_of_go, axis = 1)
    else:
        return get_top_sorted_attn(go_heatm_sc, number_of_go, axis = 0)
    
    
def load_filtered_go_id_dict(name):
    with open(name, 'rb') as fp:
        go_id_dict = pickle.load(fp)
    return go_id_dict 

    
def get_filtered_heatmap(go_heatm_sc, indexs, selected = "BOTH"):
    """ Function that removes lines and columns from heatmap
    Args:
        go_heatm_sc (numpy): heatmap of the GO terms
        indexs (list): list of indexes where the GO terms are not present in the dataset
        selected (string): "ROW" removes only rows, "COLUMN" removes only columns, "BOTH" removes both

    Returns:
    numpy: the new heatmap with the corresponding columns or rows removed
    """
    
    if selected == "BOTH":
        filt_go_heatm = go_heatm_sc[indexs]
        return filt_go_heatm[:,indexs]
    if selected == "ROW":
        return go_heatm_sc[indexs]
    if selected == "COLUMN":
        return go_heatm_sc[:,indexs]


def get_final_hm(go_heatm, number_of_go,  dim = 1):
     
    """ Function that filteres the attention matrix M to contain
    only a number of go terms with high attention, and uses
    softmax with temperature to better highlight the attention values
    Args:
        go_heatm (numpy): the attention matrix
        number_of_go (int): how many top go terms to keep
        dim (int): on what dimension to comput the sum (e.g. 1 means over the columns)
    Returns:
    numpy: filtered attention matrix
    list: the list of the indices of the selected terms
    """
    
    #retrieving the freq from the (l,l,2) matrix
    go_freq = go_heatm[:,:,1]
    #retrieving the attention sum
    go_heatmap = go_heatm[:,:,0]
    
    go_heatm_sc = go_heatmap/np.where(go_freq==0, 1e9, go_freq)
    indexs = get_filtered_indexes(go_heatm_sc, number_of_go, top = True, attn = "FOCUSED")
    
    #sort the indexes such that same go terms appear at the begining of the heatmap
    # indexs = sorted(indexs)?
    indexs = list(indexs)
    
    
    filt_go_heatm = get_filtered_heatmap(go_heatm_sc, indexs, selected = "BOTH")

    #perform softmax with temperature to better highligh the attention values
    tmpr = 0.01
    filt_go_heatm = np.where(filt_go_heatm==0, -1e9, filt_go_heatm)
    go_heatm_soft = F.softmax(torch.from_numpy(filt_go_heatm/tmpr), dim = dim)
    
    # print(len(indexs))
    
    return go_heatm_soft, indexs



def save_heatmaps(go_heatm, indexs,  filename, filename2):
    np.savez_compressed(filename, go_heatm)
    with open(filename2, 'wb') as f:
        pickle.dump(indexs, f)
    # np.savez_compressed(f'{organism}-{mode}-go_heatm_source', go_heatm_source)

def load_heatmaps(filename, filename2):
    # go_heatm_self = np.load(f'{path}{organism}-{mode}-go_heatm_self.npz')['arr_0']
    go_heatm_source = None
    if filename is not None:
        go_heatm_source = np.load(filename)['arr_0']
    with open(filename2, 'rb') as f:
        indexs = pickle.load(f)
    
    return go_heatm_source, indexs


if __name__ == '__main__':
    model_path = r'results\Sent2Vec\binary\02-06-2023-model_size-700,heads-14,nr_layers-3,size_ff-2800,dropout-0.2,epoch-20,LR-0.0001-total-model.pt'
    # go_embed_pth = "Node2Vec_Transformer_pos_neg/datasets/go-terms-64.emd"
    go_embed_pth = "Sent2Vec_Transformer_pos_neg\datasets\sentence_embedding.json"
    go_id_dict_pth = "Node2Vec_Transformer_pos_neg/datasets/go_id_dict"
    go_name_space_dict_pth = "Node2Vec_Transformer_pos_neg/datasets/go_namespace_dict"
    
    go_dic = load_filtered_go_id_dict(r'results\analysis\go_id_dict')
    # go_dic = load_filtered_go_id_dict(r'results\analysis\go_id_dict_filtered_yeast')
    
    # model = torch.load(model_path, map_location=torch.device('cpu'))
    EMB_DIM = 700
    
    confidence_score_pos = 980
    confidence_score_neg = 950
    
    # organism = [1]
    
    # # get dataset
    # train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB2(confidence_score_pos, confidence_score_neg,  go_embed_pth,
    #                                                                          go_id_dict_pth, shuffle, go_dic,  new_thingy = 'results/analysis/new_dict.pkl', organism = organism)
    # params = {'batch_size': 32, 'collate_fn': helper_collate}
    # train_grt = data.DataLoader(all_dataset, **params, shuffle = False)



    # go_heatm_self, go_heatm_source = get_heamaps(go_dic, model, train_grt)
    # print(len(go_heatm_source))
    
    # number_of_go = 30
    # act_src_hm, act_indexs = get_final_hm(go_heatm_source, number_of_go = number_of_go)
    # # act_self_hm, act_self_indexes = get_final_hm(go_heatm_self, number_of_go = number_of_go)
    
    

    # save_heatmaps(act_src_hm, act_indexs, 'results/analysis/test.npz', 'results/analysis/test_indexes.pkl')
    act_src_hm, act_indexs = load_heatmaps(None, r'results\Sent2Vec\multiclass\0_all.pkl')
    
    labels = idstogos(act_indexs, go_dic)
    expls = get_GO_explanation(labels, obo_csv_path = "results/analysis/go-basic.obo.csv")
    labels = ['('+m+')' + " " + n for m,n in zip(process_to_capital(expls['namespace']), trim_GO_expl(expls['name']))]  
    
    # print(expls)
    GO_terms = []
    for i in range(30):
        GO_terms.append(f"{expls['id'][i][2:-2]}: {expls['name'][i][2:-2]} ({expls['namespace'][i][2:-2]})")
    with open('results/analysis/Sent2vec_mul_all.txt', 'w') as f:
        f.write('\n\n'.join(GO_terms))
    
    # sns.set(font_scale=1.5)
    # fig, ax = plt.subplots(1,1, figsize=(10, 10))
    # hm = sns.heatmap(act_src_hm, ax=ax, vmax = 1, vmin = 0, square = True, cbar_kws={"shrink": 0.80}, cmap="PuBu_r") 

    # ax.tick_params(labelsize=10)
    # hm.set_xticks( np.arange(len(labels)) + 0.5) #+ 0.5 to center the label
    # hm.set_yticks( np.arange(len(labels)) + 0.5)
    # hm.set_xticklabels(labels=labels, rotation=90);
    # hm.set_yticklabels(labels=labels, rotation=0);

    # rect = fig.patch
    # rect.set_facecolor('white')
    
    # plt.show()