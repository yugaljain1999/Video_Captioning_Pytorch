import torch
from torch import nn
import torch.nn.functional as F
from attention import Attention
import numpy as np

class EncoderRNN(nn.Module):
    def __init__(self,dim_video,dim_hidden,n_layers,bidirectional,rnn_dropout=0.5,input_dropout=0.2):
        super(EncoderRNN,self).__init__()
        self.dim_video = dim_video
        self.dim_hidden = dim_hidden
        self.rnn_dropout = rnn_dropout
        self.bidirectional = bidirectional
        self.input_dropout = nn.Dropout(input_dropout)
        self.n_layers = n_layers
        self.vidhid = nn.Linear(dim_video,dim_hidden) # dimension changed
        self.out = nn.LSTM(dim_hidden,dim_hidden,num_layers=1,batch_first=True,bidirectional=self.bidirectional,dropout = self.rnn_dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.vidhid.weight)
    
        
    def forward(self,vid_features):
        batch_size , seq_len , vid_dim = vid_features.size() 
        vid_features = self.vidhid(vid_features.view(-1,vid_dim)) # vid_features -> [-1,dim_hidden]
         # vid_features is reshaped to change last dimension of video_features
        vid_features = self.input_dropout(vid_features) # vid_features -> [-1,dim_hiddem]
        vid_features = vid_features.view(batch_size,seq_len,self.dim_hidden) # vid_features -> [batch_size,seq_len,dim_hidden]
        #self.out.flatten_parameter() # transfer weights to cuda
        vid_features,hidden = self.out(vid_features) # vid_features -> [batch_size,seq_len,dim_hidden]
        #print('hidden',hidden)
        #self._init_weights() # initialize normalize weights of LSTM cell
        return vid_features,hidden

class DecoderRNN(nn.Module):
    def __init__(self,vocab_size,emb_dim,max_len,dim_hidden,n_layers,bidirectional=False,input_dropout=0.1,rnn_dropout=0.1):
        super(DecoderRNN,self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.rnn_dropout = rnn_dropout
        self.max_len = max_len
        self.bidirectional_encoder = bidirectional
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.embedding = nn.Embedding(vocab_size,emb_dim)
        self.dropout = nn.Dropout(input_dropout)
        self.attention = Attention(self.dim_hidden)

        self.rnn = nn.LSTM(self.dim_hidden + emb_dim,self.dim_hidden,num_layers=1,batch_first=True,dropout = self.rnn_dropout) # adding hidden dimensios and embedding dimensions
        self.out = nn.Linear(dim_hidden,vocab_size) # final features extracted

    def forward(self,encoder_output,encoder_hidden,mode,captions=None):
        batch_size, _ , _ = encoder_output.size()
       # print('encoder_hidden',encoder_hidden)
        decoder_hidden = self._init_hidden(encoder_hidden)
        log_probs_ls = []
        preds_ls = []
        #self.rnn.flatten_parameter() # to transfer weights to cuda
        ############ mode -> train ################
        if mode == 'train':
            target_emb = self.embedding(captions) # target_emb -> [batch_size,seq_len,emb_dim]
            for i in range(self.max_len-1):
                current_word = target_emb[:,i,:] # current target word
                # now find context using attention mechanism
                context = self.attention(decoder_hidden[0].squeeze(0),encoder_output) # video features with an attention
                decoder_input = torch.cat([current_word,context],dim=1) # concatenating encoder features and original captions
                decoder_input = self.dropout(decoder_input).unsqueeze(1) # unsqueeze decoder inputs -> []
                decoder_output,decoder_hidden = self.rnn(decoder_input,decoder_hidden)
                log_probs = F.log_softmax(self.out(decoder_output.squeeze(1)),1)
                log_probs_ls.append(log_probs.unsqueeze(1)) # list of list of probabilities 
            log_probs_ls = torch.cat(log_probs_ls,1)

        return log_probs_ls

    def _init_weights(self):
        nn.init.xavier_normal_(self.out.weight)
    def _init_hidden(self,encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden,tuple): # encoder_hidden instance is tuple only if encoder is bidirectional
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
            #print('####### tuple ')
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        #print('final_decoder_hidden',encoder_hidden)
        return encoder_hidden

    def _cat_directions(self,h):
        if self.bidirectional_encoder:
            # [layers*directions,batch_size,hid_dim] - > [layers,batch,directions*hid_dim]
            h = torch.cat([h[0:h.size(0):2],h[1:h.size(0):2]],2)
            #print('h',h)
        return h






#encoderrnn = EncoderRNN(120,100,1,True)
#decoderrnn = DecoderRNN(512,256,50,25,2)
#encoderrnn(torch.randn((32,100,180)))


