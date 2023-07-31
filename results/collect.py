import sys
# sys.path.append(r"C:\Users\\ninao\OneDrive - Delft University of Technology\Nanobiology\Bachelor End Project\Bep\Node2Vec_Transformer_pos_neg")
from evaluate import evaluate, helper_collate
from dataset_manip import get_dataset_split_stringDB, get_max_len_seq
from training_helper import transformerGO_collate_fn
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from random import shuffle
import torch.utils.data as data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def helper_collate(batch):
    MAX_LEN_SEQ = get_max_len_seq(batch)
    EMB_DIM = 64
    return transformerGO_collate_fn(batch, MAX_LEN_SEQ, EMB_DIM, pytorch_pad = False)


def helper_collate2(batch):
    MAX_LEN_SEQ = get_max_len_seq(batch)
    EMB_DIM = 700
    return transformerGO_collate_fn(batch, MAX_LEN_SEQ, EMB_DIM, pytorch_pad = False)


def picklefile(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
    print(data) 
    return data


def confusion_matrix_nee(model):
    confidence_score_pos = 850
    confidence_score_neg = 200
    EMB_DIM = 64
    model_path = model
    
    
    go_embed_pth = "Node2Vec_Transformer_pos_neg/datasets/go-terms-64.emd"
    go_id_dict_pth = "Node2Vec_Transformer_pos_neg/datasets/go_id_dict"
    go_name_space_dict_pth = "Node2Vec_Transformer_pos_neg/datasets/go_namespace_dict"
    
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    organism = [3]
    
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(confidence_score_pos, confidence_score_neg, go_id_dict_pth, 
                                                                             go_embed_pth, shuffle, "", organism = organism, go_name_space_dict_pth=go_name_space_dict_pth, 
                                                                             ratio = [0.8, 0.2, 0], stringDB = False)
    
    
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    
    c_val_grt = data.DataLoader(all_dataset, **params, shuffle = False)
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    result = evaluate(model, c_val_grt, criterion)
    
    print(result[1], result[2])
    
    print(result[3], result[5])
    con = ConfusionMatrixDisplay(confusion_matrix(result[3], result[5]))
    con.plot()
    plt.show()
    

def confusion_matrix_twee(model):
    confidence_score_pos = 990
    confidence_score_neg = 200
    EMB_DIM = 700
    model_path = model
    
    
    go_embed_pth = "Sent2Vec_Transformer_pos_neg/datasets/sentence_embedding.json"
    
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    organism = [2]
    
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(confidence_score_pos, confidence_score_neg, 
                                                                             go_embed_pth, shuffle, organism = organism,  
                                                                             ratio = [0.8, 0.2, 0])
    
    
    params = {'batch_size': 32, 'collate_fn': helper_collate2}
    
    c_val_grt = data.DataLoader(all_dataset, **params, shuffle = False)
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    result = evaluate(model, c_val_grt, criterion)
    
    print(result[1], result[2])
    
    con = ConfusionMatrixDisplay(confusion_matrix(result[3], result[5]))
    con.plot()
    plt.show()
    

def confusion_matrix_multiclass(model):
    interactions = [(1, 990), (2, 200), (9, 0)]
    model_path = model
    organism = [3]    
    
    go_embed_pth = "Node2Vec_Transformer_pos_neg/datasets/go-terms-64.emd"
    go_id_dict_pth = "Node2Vec_Transformer_pos_neg/datasets/go_id_dict"
    go_name_space_dict_pth = "Node2Vec_Transformer_pos_neg/datasets/go_namespace_dict"
    
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(interactions, go_id_dict_pth, 
                                                                             go_embed_pth, shuffle, "", organism = organism, go_name_space_dict_pth=go_name_space_dict_pth, 
                                                                             ratio = [0.8, 0, 0.2], stringDB = False)
    
    params = {'batch_size': 32, 'collate_fn': helper_collate}
    
    c_val_grt = data.DataLoader(all_dataset, **params, shuffle = False)
    criterion = nn.CrossEntropyLoss().to(device)
    
    result = evaluate(model, c_val_grt, criterion)
    
    print(result[1], result[2])
    
    print(confusion_matrix(result[3], result[5]))
    
    con = ConfusionMatrixDisplay(confusion_matrix(result[3], result[5]))
    con.plot()
    plt.show()
    

def confusion_matrix_sent2vec():
    interactions = [(1,901), (2, 700), (9, 0)]
    organism = [4]
    EMB_DIM = 700
    
    # paths
    go_embed_pth = r"Sent2Vec_Transformer_multiclass\datasets\sentence_embedding.json"
    
    # model
    model_path = r"results\Sent2Vec\multiclass\13-06-2023-Neural_network_Multiclass -total-model-0.pt"
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    train_set, valid_set, test_set, all_dataset = get_dataset_split_stringDB(interactions, go_embed_pth, shuffle, ratio = [0.8, 0, 0.2], organism = organism)
    
    params = {'batch_size': 32, 'collate_fn': helper_collate2}
    
    c_val_grt = data.DataLoader(all_dataset, **params, shuffle = False)
    criterion = nn.CrossEntropyLoss().to(device)
    
    result = evaluate(model, c_val_grt, criterion)
    
    print(result[1], result[2])
    
    print(confusion_matrix(result[3], result[5]))
    
    con = ConfusionMatrixDisplay(confusion_matrix(result[3], result[5]))
    con.plot()
    plt.show()



def graph_dropout_Node2Vec_bin():
    x_axis = [0.1, 0.2, 0.3, 0.4, 0.5]
    train = [0.959, 0.943, 0.934, 0.929, 0.920]
    val = [0.938, 0.931, 0.933, 0.912, 0.892]
    test = [0.940, 0.929, 0.926, 0.916, 0.900]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Dropout', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Node2Vec Transformer, Accuracy over Dropout', fontsize = 14)
    plt.legend()
    plt.show()
    

def graph_layers_Node2Vec_bin():
    x_axis = [2, 3, 4, 5, 6]
    train = [0.946, 0.945, 0.945, 0.936, 0.947]
    val = [0.928,  0.929, 0.929, 0.928, 0.935]
    test = [0.936, 0.936, 0.935, 0.930, 0.935]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Layers', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Node2Vec Transformer, Accuracy over Layers', fontsize = 14)
    plt.legend()
    plt.show()


def graph_heads_Node2Vec_bin():
    x_axis = [4, 8, 16]
    train = [0.934, 0.944, 0.941]
    val = [0.919, 0.921, 0.923]
    test = [0.922, 0.925, 0.920]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Heads', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Node2Vec Transformer, Accuracy over Heads', fontsize = 14)
    plt.legend()
    plt.show()
    

def graph_embedding_Node2Vec_bin():
    x_axis = [32, 64, 128]
    train = [0.896, 0.944, 0.964]
    val = [0.894, 0.921, 0.944]
    test = [0.897, 0.925, 0.940]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Embedding Dimension', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Node2Vec Transformer, Accuracy over Embedding Dimension', fontsize = 13)
    plt.legend()
    plt.show()
    

def graph_LR_Node2Vec_bin():
    x_axis = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    train = [0.850, 0.913, 0.944, 0.963, 0.959]
    val = [0.853, 0.915, 0.921, 0.940, 0.936]
    test = [0.853, 0.909, 0.925, 0.936, 0.936]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Learning Rate', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Node2Vec Transformer, Accuracy over Learning Rate', fontsize = 14)
    plt.legend()
    plt.show() 
    
    
def graph_size_Node2Vec_bin():
    x_axis = [64, 128, 256, 384, 512]
    train = [0.940, 0.940, 0.944, 0.947, 0.949]
    val = [0.931, 0.931, 0.921, 0.932, 0.934]
    test = [0.927, 0.924, 0.925, 0.932, 0.930]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Feed Forward Size', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Node2Vec Transformer, Accuracy over Feed Forward Size', fontsize = 13)
    plt.legend()
    plt.show() 
    

def graph_size_Sent2Vec_bin():
    x_axis = [700, 1400, 2800, 4200, 5600]
    train = [0.978, 0.979, 0.980, 0.977, 0.977]
    val = [0.946, 0.953, 0.952, 0.951, 0.940]
    test = [0.950, 0.948, 0.957, 0.949, 0.940]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Feed Forward Size', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Sent2Vec Transformer, Accuracy over Feed Forward Size', fontsize = 14)
    plt.legend()
    plt.show() 
 
    
def graph_dropout_Sent2Vec_bin():
    x_axis = [0.1, 0.2, 0.3, 0.4, 0.5]
    train = [0.978, 0.980, 0.978, 0.979, 0.977]
    val = [0.947, 0.952, 0.950, 0.947, 0.943]    
    test = [0.948, 0.957, 0.950, 0.949, 0.939]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Dropout', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Sent2Vec Transformer, Accuracy over Dropout', fontsize = 14)
    plt.legend()
    plt.show()
  
    
def graph_heads_Sent2Vec_bin():
    x_axis = [4, 7, 14, 28, 35]
    train = [0.947, 0.972, 0.980, 0.979, 0.983]
    val = [0.929, 0.937, 0.952, 0.953, 0.950]
    test = [0.92, 0.939, 0.957, 0.953, 0.954]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Heads', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Sent2Vec Transformer, Accuracy over Heads', fontsize = 14)
    plt.legend()
    plt.show()
    
    
def graph_layers_Sent2Vec_bin():
    x_axis = [2, 3, 4, 5, 6]
    train = [0.981, 0.980, 0.975, 0.976, 0.980]
    val = [0.953, 0.952, 0.946, 0.951, 0.951]
    test = [0.950, 0.957, 0.940, 0.950, 0.949]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Layers', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Sent2Vec Transformer, Accuracy over Layers', fontsize = 14)
    plt.legend()
    plt.show()
    

def graph_learning_rate_Sent2Vec_bin():
    x_axis = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    train = [0.985, 0.984, 0.980, 0.915, 0.893]
    val = [0.951, 0.953, 0.952, 0.912, 0.886]
    test = [0.957, 0.953, 0.957, 0.920, 0.897]
    
    plt.plot(x_axis, train, label = 'train', marker='o')
    plt.plot(x_axis, val, label = 'val', marker='o')
    plt.plot(x_axis, test, label = 'test', marker='o')
    plt.xlabel('Learning Rate', fontsize = 12)
    plt.ylabel('Accuracy', fontsize = 12)
    plt.title('Binary Sent2Vec Transformer, Accuracy over Learning Rate', fontsize = 14)
    plt.legend()
    plt.show()
    
    
def create_conf_matrix():
    matr = np.array([[13342, 2779, 1953], [806, 12121, 3247], [1876, 3375, 11840]])
    con = ConfusionMatrixDisplay(matr)
    con.plot()
    plt.show()
    

def binary_RNN_epoch_loss_acc():
    # with open(r"results\RNN\binary\13-06-2023-simple_rnn_result_gpu_legit.pkl", 'rb') as f:
    #     data = pkl.load(f)
    # print(np.average(data['train_acc']))
    # print(np.average(data['val_acc'])) 
    # print(np.average(data['test_acc']))
    # print(np.average(data['train_roc']))
    # print(np.average(data['val_roc']))
    # print(np.average(data['test_roc']))
    train = np.zeros(20)
    val = np.zeros(20)
    for i in range(5):
        with open(f"results/RNN/binary/14-06-2023-metrics_callback-{i}.pkl", 'rb') as f:
            data = pkl.load(f)
        # print(data)
        train += np.array(data['acc'])
        val += np.array(data['val_acc'])
    # print(train/5)
    plt.plot(np.arange(20), train/5, label = 'train', marker = 'o')
    plt.plot(np.arange(20), val/5, label = 'val', marker = 'o')
    plt.xlabel('Epoch', fontsize = 13)
    plt.ylabel('Accuracy', fontsize = 13)
    plt.title('Binary RNN, Accuracy over Epochs', fontsize  = 12)
    plt.legend()
    plt.show()
    

def create_sent2vec_plot():
    accuracy = [0.826743338857161, 0.8828810062705045, 0.9086271883768328, 0.9230777524304732, 0.9349711215096798, 0.9441277022567092, 0.9526972539687271, 0.9540643078858773, 0.9589883824854947, 0.960993394897315, 0.9631351127008502, 0.967780415521284, 0.9705118428577076, 0.9726589216569964, 0.9729752204064547, 0.975344780529515, 0.9743878427875099, 0.9765322410889219, 0.9774864983330503, 0.9781271373255972]
    loss = [0.3661607764780141, 0.2573902027553729, 0.20486111626433035, 0.1750388314629618, 0.1462256122392378, 0.13027283626696187, 0.11280881579032545, 0.10913179260886541, 0.09971105889974242, 0.09233960556014897, 0.08727849821524436, 0.08147019716711236, 0.07497571723199965, 0.07013019294540385, 0.06898427700146326, 0.0646051985855111, 0.06491726384913338, 0.060478195383646446, 0.05700812267976556, 0.053952497493737844]
    accuracy_val = [0.8345162367365163, 0.8905065276060894, 0.9135577849521759, 0.9166515063328348, 0.9233848999260338, 0.9292083754660977, 0.9380649111832783, 0.926721266120862, 0.9373976379443126, 0.9377009439620243, 0.9403093757143446, 0.9428571462631226, 0.9432211134843765, 0.9426751626524955, 0.9484379769890172, 0.9432817746879189, 0.9452229332012735, 0.9420685506170723, 0.9437670643162576, 0.9449802883871042]
    loss_val = [0.3624191509120783, 0.23310736590509962, 0.19844623261196598, 0.19698030782780448, 0.1775491150796034, 0.18936685326799846, 0.17504190243068773, 0.18367263543045825, 0.18397790352831694, 0.16977771242878809, 0.1653221866566759, 0.15904982541064927, 0.17462638174154008, 0.1772432373480717, 0.1693394892095665, 0.21350448257897855, 0.1846564082357629, 0.18272086439630503, 0.20886165200833492, 0.18032465966501435]
    accuracy2 = [0.827509961107321, 0.8839746494042245, 0.9064452631051461, 0.9187943168234026, 0.9298218850884141, 0.9356546484682549, 0.9411255446347323, 0.9459986897746911, 0.9514669054432919, 0.9552919759134356, 0.9579376273178027, 0.9604465733304548, 0.96240601727837, 0.9655502412878155, 0.9686435358376024, 0.9713830046676563, 0.9699167723290658, 0.9709273200286063, 0.9740688635401749, 0.9759398510581568]
    loss2 = [0.370585174105194, 0.2578881008511905, 0.21540201996620953, 0.18732538851991035, 0.16801543456860232, 0.15403865569641004, 0.14273182793922926, 0.13245641229008778, 0.12009472888485904, 0.10960968483411715, 0.10380761446862034, 0.09873997390439565, 0.09216135720896165, 0.08725797432397553, 0.0821602132930413, 0.07415768945880422, 0.0751774930532582, 0.07008598338399659, 0.06837344545617766, 0.06253674750044003]
    accuracy_val2 = [0.8795875109684695, 0.8730361009858976, 0.9066424077483499, 0.9141037357840568, 0.9228389490941528, 0.9312101951829946, 0.9308462279617407, 0.9395814412718366, 0.9301182935192327, 0.9430391298737496, 0.9408553265462256, 0.9441310315375115, 0.9441310315375115, 0.9432211134843765, 0.9455869004225276, 0.9429178074666649, 0.9499545070775754, 0.9510464087413375, 0.9495905398563215, 0.9510464087413375]
    loss_val2 = [0.2699073741485359, 0.279850295299937, 0.2104997278256401, 0.20574360524725382, 0.1914723784081448, 0.17300420941393466, 0.16525429255880747, 0.15655651532267784, 0.16402855414932796, 0.14747773384924528, 0.16600226303139867, 0.14851113805950733, 0.14243511039953513, 0.17315531717642998, 0.15472962923205583, 0.14374159568932596, 0.14737318036484587, 0.1576199654288068, 0.17289453602891272, 0.154590589796626]
    # plt.plot(np.arange(20), (np.array(accuracy)+np.array(accuracy2))/2, label = 'train', marker = 'o')
    # plt.plot(np.arange(20), (np.array(accuracy_val)+np.array(accuracy_val2))/2, label = 'val', marker = 'o')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Binary Sent2Vec, Accuracy over Epochs')
    # plt.legend()
    # plt.show()
    # plt.clear()
    plt.plot(np.arange(20), (np.array(loss)+np.array(loss2))/2, label = 'train', marker = 'o')
    plt.plot(np.arange(20), (np.array(loss_val)+np.array(loss_val2))/2, label = 'val', marker = 'o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Binary Sent2Vec, Loss over Epochs')
    plt.show()  
    
    
def averages_fun(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
    # print(data)
    print(np.mean(data['final_test'], axis=0))
    print(np.mean(data['train_accs']))
    print(np.mean(data['val_accs']))
    print(np.mean(data['train_rocs']))
    print(np.mean(data['val_rocs']))


def multiclass_embedding():
    x_axis = [32,64,128]
    train = [0.789,0.866, 0.926]
    val = [0.781,0.841,0.881]
    test = [0.786,0.847,0.880]
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Embedding Size", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Node2Vec, Accuracy over Embedding Size", fontsize = 14)
    plt.legend()
    plt.show()

def multiclass_heads():
    x_axis = [4,8,16]
    train = [0.860,0.878,0.878]
    val = [0.845,0.854,0.854]
    test = [0.849,0.858,0.858]
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Number of Heads", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Node2Vec, Accuracy over Number of Heads", fontsize = 14)
    plt.legend()
    plt.show()
    
    
def multiclass_layers():
    x_axis = [2,3,4,5,6]
    train = [0.878, 0.878, 0.876, 0.870, 0.853]
    val = [0.856, 0.854, 0.837, 0.863, 0.822]
    test = [0.854, 0.858, 0.844, 0.858, 0.815]
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Number of Layers", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Node2Vec, Accuracy over Number of Layers", fontsize = 14)
    plt.legend()
    plt.show()
    
    
def multiclass_LR():
    x_axis = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    train = [0.696,0.814,0.866,0.919,0.904]
    val = [0.658,0.780,0.841,0.889,0.877]
    test = [0.640,0.788,0.847,0.886,0.879]
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Learning Rate", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Node2Vec, Accuracy over Learning Rate", fontsize = 14)
    plt.legend()
    plt.show()
    
    
def multiclass_size():
    x_axis = [64, 128, 256, 384, 512]
    train = [0.862,0.872,0.878,0.882,0.888]
    val = [0.833,0.849,0.854,0.869,0.873]
    test = [0.832,0.845,0.858,0.864,0.867]
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Feed Forward Size", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Node2Vec, Accuracy over Feed Forward Size", fontsize = 14)
    plt.legend()
    plt.show()
    
    
def multiclass_dropout():
    x_axis = [0.1,0.2,0.3,0.4,0.5]
    train = [0.906, 0.878, 0.852, 0.832, 0.819]
    val = [0.871,0.854, 0.844, 0.830, 0.798]
    test = [0.871,0.858,0.847, 0.830, 0.799]
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Dropout", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Node2Vec, Accuracy over Dropout", fontsize = 14)
    plt.legend()
    plt.show()
    

def sent2vec_multiclass_heads():
    x_axis = [4, 7, 14, 28, 35, 70]
    train = [0.892, 0.943, 0.960, 0.967, 0.969, 0.971]
    val = [0.873, 0.898, 0.897, 0.904, 0.911, 0.912]
    test = [0.865, 0.889, 0.901, 0.901, 0.910, 0.914]
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Heads", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Sent2Vec, Accuracy over Heads", fontsize = 14)
    plt.legend()
    plt.show()
    

def sent2vec_multiclass_layers():
    x_axis = [2, 3, 4, 5, 6]
    train = [0.963, 0.951, 0.960, 0.957, 0.956]
    val = [0.910, 0.893, 0.913, 0.898, 0.895]
    test = [0.907, 0.897, 0.906, 0.891, 0.897]
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Layers", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Sent2Vec, Accuracy over Layers", fontsize = 14)
    plt.legend()
    plt.show()
    
    
def sent2vec_multiclass_LR():
    x_axis = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    train = [0.969, 0.968, 0.960, 0.837, 0.760]
    val = [0.901, 0.903, 0.913, 0.817, 0.738]
    test = [0.900, 0.907, 0.904, 0.808, 0.731]
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Learning Rate", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Sent2Vec, Accuracy over Learning Rate", fontsize = 14)
    plt.legend()
    plt.show()
    

def sent2vec_multiclass_size():
    x_axis = [700, 1400, 2800, 4200, 5600]
    train = [0.958, 0.956, 0.958, 0.957, 0.960]
    val = [0.895, 0.9, 0.902, 0.899, 0.906]
    test = [0.897, 0.901, 0.900, 0.896, 0.900]
    
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Feed Forward Size", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Sent2Vec, Accuracy over Feed Forward Size", fontsize = 14)
    plt.legend()
    plt.show()
    

def sent2vec_multiclass_dropout():
    x_axis = [0.1, 0.2, 0.3, 0.4, 0.5]
    train = [0.961, 0.960, 0.953, 0.955, 0.946]
    val = [0.899, 0.913, 0.896, 0.895, 0.873]
    test = [0.902, 0.904, 0.889, 0.901, 0.880]
    
    plt.plot(x_axis, train, label = 'train', marker = 'o')
    plt.plot(x_axis, val, label = 'val', marker = 'o')
    plt.plot(x_axis, test, label = 'test', marker = 'o')
    plt.xlabel("Dropout", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.title("Multiclass Sent2Vec, Accuracy over Dropout", fontsize = 14)
    plt.legend()
    plt.show()

    
def barplot():
    node_bin_train_acc = np.array([0.9447945506903377, 0.9447945506903377, 0.9434212182560107, 0.9447536496350365, 0.9414689781021898])
    node_mul_train_acc = np.array([0.87372201, 0.87484641, 0.87557524, 0.88251437, 0.88015545])
    sent_bin_train_acc = np.array([0.9774891788309271, 0.9810435190155176, 0.9725222162652815, 0.9800865812735124, 0.9804966974486574])
    sent_mul_train_acc = np.array([0.9656449763757643, 0.9655146956642578, 0.9639426417454141, 0.964307427737632, 0.9620926556420234])
    rnn_bin_train_acc = np.array([0.8883616924285889, 0.890734076499939, 0.9025502800941467, 0.9135909676551819, 0.9070212841033936])
    rnn_mul_train_acc = np.array([0.7847315073013306, 0.7753865718841553, 0.7799220681190491, 0.7961463332176208, 0.7805917263031006])
    
    node_bin_train_roc = np.array([0.9884481646683303, 0.9896740605667582, 0.9887280108684453, 0.9894545547501056, 0.98816247069534])
    node_mul_train_roc = np.array([0.9711657528162801, 0.9712771437150186, 0.9719064902228526, 0.9746707287992623, 0.9741633979907552])
    sent_bin_train_roc = np.array([0.9976800887761573, 0.9981776666236342, 0.9965185270641302, 0.9980053110972233, 0.9984524004272478])
    sent_mul_train_roc = np.array([0.9969577921549546, 0.9966662448617, 0.9968507407676582, 0.996597458229782, 0.9966342651255671])
    rnn_bin_train_roc = np.array([0.9658727648012747, 0.9646018527889235, 0.9728062835314023, 0.9771270777514843, 0.9755103822004926])
    rnn_mul_train_roc = np.array([0.9294192420757496, 0.9264220548478508, 0.9262919514379581, 0.9345398061482343, 0.9271988902980134])
    
    
    node_bin_val_acc = np.array([0.9332174007282701, 0.9285973837209303, 0.9324127906976745, 0.9333212209302325, 0.9367732558139535])
    node_mul_val_acc = np.array([0.861132342041875, 0.8597368908652956, 0.8501250694830461, 0.8587525477116916, 0.8650233926255326])
    sent_bin_val_acc = np.array([0.9523809552192688, 0.9474673977323399, 0.9458295452366968, 0.9512890535555069, 0.9524416164228111])
    sent_mul_val_acc = np.array([0.9027966926070039, 0.9049043450064851, 0.9092817769130999, 0.9143806744487678, 0.90878728923476])
    rnn_bin_val_acc = np.array([0.8359189629554749, 0.8439496159553528, 0.8397517800331116, 0.8545355200767517, 0.8541704416275024])
    rnn_mul_val_acc = np.array([0.7202873826026917, 0.7215051054954529, 0.7179737091064453, 0.7219921946525574, 0.7184607982635498])
    
    
    node_bin_val_roc = np.array([0.9847505907746872, 0.9814849322858477, 0.9820659173188042, 0.983476925131904, 0.9828376913906458])
    node_mul_val_roc = np.array([0.9650847830646314, 0.9651990463557018, 0.9622995540938465, 0.9639549132962865, 0.9681019283245611])
    sent_bin_val_roc = np.array([0.9897244145037888, 0.988839039758656, 0.9876365727429557, 0.9892462987024773, 0.9891006080242087])
    sent_mul_val_roc = np.array([0.9781250944905083, 0.9779036408079599, 0.9767921371201737, 0.9799580341976144, 0.9800774122677708])
    rnn_bin_val_roc = np.array([0.9260200336242846, 0.9272943264594782, 0.9328897838551902, 0.9392176962416823, 0.9334615039849299])
    rnn_mul_val_roc = np.array([0.8878716845164106, 0.8867970368536812, 0.885631823256007, 0.8845310573631977, 0.8863856818291955])
    
    
    node_bin_test_acc = np.array([0.9264534883720931, 0.9338662790697675, 0.9219476744186047, 0.9263081395348837, 0.9332848837209302])
    node_mul_test_acc = np.array([0.861803392177224, 0.8540945540556132, 0.8472871235721704, 0.8581653109495789, 0.8640677281643014])
    sent_bin_test_acc = np.array([0.9492279482739312, 0.9519976274091371, 0.941896126890669, 0.9511229918927563, 0.9506856741345658])
    sent_mul_test_acc = np.array([0.9005656610470274, 0.8982780612244898, 0.9027978039041703, 0.9028393966282166, 0.906388642413487])
    rnn_bin_test_acc = np.array([0.8303649425506592, 0.8486131429672241, 0.8335766196250916, 0.8582481741905212, 0.8467153310775757])
    rnn_mul_test_acc = np.array([0.7195870280265808, 0.7220219969749451, 0.7197818160057068, 0.7286451458930969, 0.7209506034851074])
    
    
    node_bin_test_roc = np.array([0.9831584846785754, 0.9838405980671833, 0.981757031535428, 0.9832963759193567, 0.9831469581661881])
    node_mul_test_roc = np.array([0.9651212825882903, 0.9624004515635542, 0.9626625890190313, 0.9653223849118212, 0.9671593865815232])
    sent_bin_test_roc = np.array([0.9858890372574212, 0.9899387755032363, 0.9861338396452175, 0.9886427226942057, 0.9885758032130231])
    sent_mul_test_roc = np.array([0.9759269201181104, 0.9756999045454898, 0.9771930366717155, 0.9764073510078499, 0.9786773915510913])
    rnn_bin_test_roc = np.array([0.9232034248695866, 0.9307909716643665, 0.9271455519212608, 0.942048393508324, 0.9285804339284637])
    rnn_mul_test_roc = np.array([0.8864883308195587, 0.8849881809376029, 0.8870579640134671, 0.8837336129592647, 0.8819400739132316])
    
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    train_acc = [np.mean(node_bin_train_acc), np.mean(node_mul_train_acc), np.mean(sent_bin_train_acc), np.mean(sent_mul_train_acc), np.mean(rnn_bin_train_acc), np.mean(rnn_mul_train_acc)]
    val_acc = [np.mean(node_bin_val_acc), np.mean(node_mul_val_acc), np.mean(sent_bin_val_acc), np.mean(sent_mul_val_acc), np.mean(rnn_bin_val_acc), np.mean(rnn_mul_val_acc)]
    test_acc = [np.mean(node_bin_test_acc), np.mean(node_mul_test_acc), np.mean(sent_bin_test_acc), np.mean(sent_mul_test_acc), np.mean(rnn_bin_test_acc), np.mean(rnn_mul_test_acc)]
    train_error = [np.std(node_bin_train_acc), np.std(node_mul_train_acc), np.std(sent_bin_train_acc), np.std(sent_mul_train_acc), np.std(rnn_bin_train_acc), np.std(rnn_mul_train_acc)]
    val_error = [np.std(node_bin_val_acc), np.std(node_mul_val_acc), np.std(sent_bin_val_acc), np.std(sent_mul_val_acc), np.std(rnn_bin_val_acc), np.std(rnn_mul_val_acc)]
    test_error = [np.std(node_bin_test_acc), np.std(node_mul_test_acc), np.std(sent_bin_test_acc), np.std(sent_mul_test_acc), np.std(rnn_bin_test_acc), np.std(rnn_mul_test_acc)]
    
    train_roc = [np.mean(node_bin_train_roc), np.mean(node_mul_train_roc), np.mean(sent_bin_train_roc), np.mean(sent_mul_train_roc), np.mean(rnn_bin_train_roc), np.mean(rnn_mul_train_roc)]
    val_roc = [np.mean(node_bin_val_roc), np.mean(node_mul_val_roc), np.mean(sent_bin_val_roc), np.mean(sent_mul_val_roc), np.mean(rnn_bin_val_roc), np.mean(rnn_mul_val_roc)]
    test_roc = [np.mean(node_bin_test_roc), np.mean(node_mul_test_roc), np.mean(sent_bin_test_roc), np.mean(sent_mul_test_roc), np.mean(rnn_bin_test_roc), np.mean(rnn_mul_test_roc)]
    train_roc_error = [np.std(node_bin_train_roc), np.std(node_mul_train_roc), np.std(sent_bin_train_roc), np.std(sent_mul_train_roc), np.std(rnn_bin_train_roc), np.std(rnn_mul_train_roc)]
    val_roc_error = [np.std(node_bin_val_roc), np.std(node_mul_val_roc), np.std(sent_bin_val_roc), np.std(sent_mul_val_roc), np.std(rnn_bin_val_roc), np.std(rnn_mul_val_roc)]
    test_roc_error = [np.std(node_bin_test_roc), np.std(node_mul_test_roc), np.std(sent_bin_test_roc), np.std(sent_mul_test_roc), np.std(rnn_bin_test_roc), np.std(rnn_mul_test_roc)]
    
    br1 = np.arange(len(train_acc))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    plt.bar(br1, train_acc, yerr=train_error, width = barWidth, label = "Train Accuracy", align='center', ecolor='black', capsize=5)
    plt.bar(br2, val_acc, yerr=val_error, width = barWidth, label = "Validation Accuracy", align='center', ecolor='black', capsize=5)
    plt.bar(br3, test_acc, yerr=test_error, width = barWidth, label = "Test Accuracy", align='center', ecolor='black', capsize=5)
    plt.xticks([r + barWidth for r in range(len(train_acc))], ['Node2Vec Binary', 'Node2Vec Multiclass', 'Sent2Vec Binary', 'Sent2Vec Multiclass', 'RNN Binary', 'RNN Multiclass'], fontsize = 13.5)
    
    plt.title('Accuracy of Models', fontsize=20)
    plt.grid(axis='y')  # Add y-axis grid
    plt.ylabel('Accuracy', fontsize = 18)
    plt.legend()
    plt.show()
    
    plt.bar(br1, train_roc, yerr=train_error, width = barWidth, label = "Train Accuracy", align='center', ecolor='black', capsize=5)
    plt.bar(br2, val_roc, yerr=val_error, width = barWidth, label = "Validation Accuracy", align='center', ecolor='black', capsize=5)
    plt.bar(br3, test_roc, yerr=test_error, width = barWidth, label = "Test Accuracy", align='center', ecolor='black', capsize=5)
    plt.xticks([r + barWidth for r in range(len(train_acc))], ['Node2Vec Binary', 'Node2Vec Multiclass', 'Sent2Vec Binary', 'Sent2Vec Multiclass', 'RNN Binary', 'RNN Multiclass'], fontsize = 13.5)
    
    plt.title('ROC AUC of Models', fontsize=20)
    plt.grid(axis='y')  # Add y-axis grid
    plt.ylabel('ROC AUC', fontsize = 18)
    plt.legend()
    plt.show()
    
    
    
def create_new_plot():
    node_mul = [0.61667014, 0.65461483, 0.68374907, 0.7110837, 0.74372105, 0.76511372, 0.78425283, 0.79562373, 0.80286965, 0.81581203, 0.81978414, 0.8207708,
                0.83195873, 0.83392741, 0.84162498, 0.84744766, 0.8513885, 0.85492751, 0.85805309, 0.85743469]
    node_bin = [0.80023879, 0.84493875, 0.86695909, 0.87595515, 0.88253218, 0.88520037, 0.89138289, 0.89625208, 0.89857766, 0.90339493, 0.90728302, 0.90944767,
                0.91596761, 0.91893169, 0.92149086, 0.9195598, 0.91936773, 0.9262147, 0.92755918, 0.9315407]
    sent_mul = [0.7473022, 0.81926556, 0.8485976, 0.8686965, 0.87666991, 0.87983301, 0.88723087, 0.88786479, 0.88929799, 0.89196498, 0.89404183, 0.89858463,
                0.89885538, 0.89980058, 0.90058204, 0.89898184, 0.90342899, 0.90392023, 0.90296855, 0.90462062]
    sent_bin = [0.86721263, 0.89819837, 0.90926297, 0.91637246, 0.92725508, 0.92856537, 0.93346679, 0.93649985, 0.93825903, 0.94094025, 0.9403579, 0.93808918,
                0.93929027, 0.94516227, 0.9455869, 0.94619351, 0.94878981, 0.94557477, 0.94694571, 0.9474674]
    rnn_bin = [0.81452819, 0.82595365, 0.82927542, 0.83686805, 0.83916773, 0.83967876, 0.84446068, 0.84763643, 0.84756343, 0.85008214, 0.84679685, 0.84453368,
                0.85165176, 0.84946158, 0.84971712, 0.85125023, 0.85409745, 0.85095822, 0.84916955, 0.84566526]
    rnn_mul = [0.60333577, 0.63199415, 0.65514976, 0.66725104, 0.67762357, 0.68672997, 0.69082055, 0.69712687, 0.70051131, 0.70204529, 0.70462624, 0.71246651,
                0.71032383, 0.71158998, 0.71139517, 0.71770148, 0.71979547, 0.72132945, 0.72388604, 0.72369125]
    
    plt.plot(np.arange(20), node_bin, label='Node2Vec Binary', marker = 'o')
    plt.plot(np.arange(20), node_mul, label='Node2Vec Multiclass', marker = 'o')
    plt.plot(np.arange(20), sent_bin, label='Sent2Vec Binary', marker = 'o')
    plt.plot(np.arange(20), sent_mul, label='Sent2Vec Multiclass', marker = 'o')
    plt.plot(np.arange(20), rnn_bin, label='RNN Binary', marker = 'o')
    plt.plot(np.arange(20), rnn_mul, label='RNN Multiclass', marker = 'o')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation accuracy over epoch of all models')
    plt.show()
    

def create_new_training_plot():
    node_bin = [0.77032249, 0.83026208, 0.85482601, 0.86855336, 0.87930154, 0.88660993,
             0.8917276,  0.8972505,  0.90374151, 0.90838818, 0.91294582, 0.91752234,
            0.92157721, 0.92556443, 0.92803077, 0.93176787, 0.93553077, 0.93850145,
            0.94083816, 0.94384659]
    node_mul = [0.57352408, 0.64321098, 0.67900873, 0.70600541, 0.7332206,  0.75652357,
                0.77486119, 0.79140672, 0.80254063, 0.81231192, 0.82258398, 0.83038351,
                0.83789064, 0.84453882, 0.85083751, 0.85832928, 0.86393767, 0.86833183,
                0.87356175, 0.8773627 ]
    sent_bin = [0.82694223, 0.88621287, 0.90733036, 0.9228869,  0.93402276, 0.9417056,
                0.94661092, 0.95276212, 0.95553161, 0.95876697, 0.96183936, 0.96493802,
                0.96648734, 0.96857492, 0.97130956, 0.97293126, 0.97481833, 0.9756659,
                0.97585729, 0.97832764]
    sent_mul = [0.67819361, 0.79905069, 0.85148694, 0.87715658, 0.89348336, 0.90649145,
                0.91561197, 0.92304753, 0.93038494, 0.93580809, 0.93957667, 0.94455079,
                 0.94782171, 0.95108046, 0.9532162,  0.95702474, 0.95808956, 0.95993521,
                0.96247655, 0.96430048]
    rnn_bin = [0.74715086, 0.82209042, 0.8326931,  0.83959122, 0.84513892, 0.84973768,
                0.85308636, 0.85823258, 0.86083307, 0.86470186, 0.86874402, 0.87134449,
                0.87491218, 0.87825174, 0.88130846, 0.88535974, 0.88918289, 0.8904512,
                0.89468498, 0.8979333 ]
    rnn_mul = [0.5252503,  0.62941661, 0.65598466, 0.67271068, 0.68543168, 0.69487813,
            0.70275419, 0.71131806, 0.71871938, 0.72523204, 0.72949878, 0.73564016,
            0.74002253, 0.74542744, 0.75029063, 0.7539426,  0.75827019, 0.76178216,
            0.76612191, 0.76975563]
    plt.plot(np.arange(20), node_bin, label='Node2Vec Binary', marker = 'o')
    plt.plot(np.arange(20), node_mul, label='Node2Vec Multiclass', marker = 'o')
    plt.plot(np.arange(20), sent_bin, label='Sent2Vec Binary', marker = 'o')
    plt.plot(np.arange(20), sent_mul, label='Sent2Vec Multiclass', marker = 'o')
    plt.plot(np.arange(20), rnn_bin, label='RNN Binary', marker = 'o')
    plt.plot(np.arange(20), rnn_mul, label='RNN Multiclass', marker = 'o')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training accuracy over epoch of all models')
    plt.show()
    

def another_barplot():
    activating = [6.83, 2.68, 26.1]
    inhibiting = [8.27, 5.5, 25.1]
    neutral = [24.3, 12.8, 30.7] 
    
    barWidth = 0.25
    fig = plt.subplots(figsize =(8, 6))
    
    br1 = np.arange(len(activating))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    plt.bar(br1, activating, label='Activating', width = barWidth)
    plt.bar(br2, inhibiting, label='Inhibiting', width = barWidth)
    plt.bar(br3, neutral, label='Neutral', width = barWidth)
    plt.xticks([r + barWidth for r in range(len(activating))], ['Node2Vec', 'Sent2Vec', 'RNN'])
    plt.title('Percentage of misclassified activating, inhibiting, and neutral interactions')
    plt.grid(axis='y')  # Add y-axis grid
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()
    
    
def and_another_one():
    total = [0.93, 0.87, 0.98, 0.93]
    human = [0.95, 0.86, 0.98, 0.93]
    yeast = [0.93, 0.83, 0.98, 0.93]
    pombe = [0.87, 0.70, 0.90, 0.65]
    coli = [0.21, 0.32, 0.34, 0.36]
    
    # node_acc = [0.93, 0.95, 0.93, 0.87, 0.21]
    # node_roc = [0.98, 0.99, 0.97, 0.94, 0.30]
    # sent_acc = [0.98, 0.98, 0.98, 0.90, 0.34]
    # sent_roc = [0.99, 0.99, 0.99, 0.96, 0.29]
    
    barWidth = 0.125
    fig = plt.subplots(figsize =(12, 8))
    
    br1 = np.arange(len(total))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    
    plt.bar(br1, total, label='Total', width = barWidth)
    plt.bar(br2, human, label='H. sapiens', width = barWidth)
    plt.bar(br3, yeast, label='S. cerevisiae', width = barWidth)
    plt.bar(br4, pombe, label='S. pombe', width = barWidth)
    plt.bar(br5, coli, label='E. coli', width = barWidth)
    plt.xticks([r + barWidth+barWidth for r in range(len(total))], ['Node2Vec Binary', 'Node2Vec Multiclass', 'Sent2Vec Binary', 'Sent2Vec Multiclass'], fontsize = 16)
    plt.title("Accuracy of Node2Vec and Sent2Vec Transformers on different species", fontsize = 18)
    plt.grid(axis='y')  # Add y-axis grid
    plt.legend(loc='lower left', fontsize = 14)
    plt.show()  
    
  
def barplot_500():
    bio = [20, 13, 7, 4]
    mol = [6, 7, 7, 8]
    cell = [4, 10, 16, 14]
    
    barWidth = 0.20
    fig = plt.subplots(figsize =(12, 8))
    
    br1 = np.arange(len(bio))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    
    plt.bar(br1, bio, label='Biological Process', width = barWidth)
    plt.bar(br2, mol, label='Molecular Function', width = barWidth)
    plt.bar(br3, cell, label='Cellular Component', width = barWidth)
    plt.xticks([r + barWidth*1 for r in range(len(bio))], ['Binary Node2Vec', 'Multiclass Node2Vec', 'Binary Sent2Vec', 'Multiclass Sent2Vec'], fontsize = 16)
    plt.title("Count of GO term type for the 30 most important GO terms for each model", fontsize = 18)
    plt.grid(axis='y')  # Add y-axis grid
    plt.legend()
    plt.ylabel("Count of GO terms", fontsize = 16)
    plt.show()
      

def plot_acc(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
    print(data.keys()) 
    train = np.zeros(20)
    val = np.zeros(20)
    for i in range(5):
        train += np.array(data['train_epoch'][i]['acc'])
        val += np.array(data['val_epoch'][i]['acc'])
    plt.plot(range(1, 21), train/5, label = 'train', marker='o')
    plt.plot(range(1, 21), val/5, label = 'val', marker='o')
    plt.xlabel('Epoch', fontsize = 13)
    plt.ylabel('Accuracy', fontsize = 13)
    plt.title('Multiclass Sent2Vec Transformer, Accuracy over Epochs', fontsize = 12)
    plt.legend()
    plt.show()
    # print(train/10)
    return data

def plot_loss(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
    print(data.keys()) 
    train = np.zeros(20)
    val = np.zeros(20)
    for i in range(5):
        train += np.array(data['train_epoch'][i]['loss'])
        val += np.array(data['val_epoch'][i]['loss'])
    plt.plot(range(1, 21), train/5, label = 'train', marker='o')
    plt.plot(range(1, 21), val/5, label = 'val', marker='o')
    plt.xlabel('Epoch', fontsize = 13)
    plt.ylabel('Loss', fontsize = 13)
    plt.title('Multiclass Sent2Vec Transformer, Loss over Epochs', fontsize = 12)
    plt.legend()
    plt.show()
    # print(train/10)
    return data
    

if __name__ == '__main__':
    pass