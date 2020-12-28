######### Attention Mechanism on output features from decoder #######

import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,dim):
        super(Attention,self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim*2,dim)
        self.linear2 = nn.Linear(dim,1)

    def _init_hidden_(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self,decoder_hidden,encoder_outputs):
        # decoder_hidden - > [batch_size,dim]
        # encoder_outputs -> [batch_size,seq_len,dim]
        # decoder_hidden.unsqueeze(1).repeat(1,seq_len,1) -> [batch_size,seq_len,dim] REMEMBER here repeat function is used to repeat dim=2 seq_len times to match with second dimension of encoder outputs so that concatenation is possible
        # after concatenating at dim = 2 -> [batch_size,seq_len,dim*2] 
        batch_size,seq_len,_ = encoder_outputs.size()
        context = torch.cat((decoder_hidden.unsqueeze(1).repeat(1,seq_len,1),encoder_outputs),dim = 2) # context [batch_size,seq_len,dim*2]
        context = context.view(-1,self.dim*2) # reshape tensors as next layer is linear which takes input of dimensions=2
        out = self.linear2(torch.tanh(self.linear1(context))) # out -> [-1,1]
        out = out.view(batch_size,seq_len) # out -> [batch_size,seq_len]
        alpha = F.softmax(out,dim=1) # alpha -> [batch_size,seq_len] alpha.unsqueeze(1) -> [batch_size,1,seq_len]
                                    #                                encoder_outputs -> [batch_size,seq_len,dim]
        attention = torch.bmm(alpha.unsqueeze(1),encoder_outputs) # attention -> [batch_size,1,dim]
        attention = attention.squeeze(1) # [batch_size,dim]
        context = attention
        return context

