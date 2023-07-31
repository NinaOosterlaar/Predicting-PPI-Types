from harvard_transformer import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TransformerGO_Scratch(nn.Module):
    def __init__(self ,d_model, nhead, num_layers, dim_feedforward, dropout = 0.1):
        super().__init__()
        
        c = copy.deepcopy
        attn = MultiHeadedAttention(nhead, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), num_layers)

        self.linear = nn.Linear(d_model, 1)
    
    #batch  * max_seq_len * node2vec_dim
    def forward(self, emb_proteinA, emb_proteinB, protA_mask, protB_mask):
        
        memory = self.encoder(emb_proteinA, protA_mask)
        output = self.decoder(emb_proteinB, memory, protA_mask, protB_mask)
        #output: batch * seqLen * embDim
        
        #transform B * seqLen * node2vec_dim --> B * node2vec_dim (TransformerCPI paper)
        output_c = torch.linalg.norm(output, dim = 2)
        output_c = F.softmax(output_c, dim = 1).unsqueeze(1)
        output = torch.bmm(output_c, output)
        
        return self.linear(output).squeeze(1)
    

