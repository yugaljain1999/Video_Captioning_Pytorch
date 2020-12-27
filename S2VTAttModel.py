import torch
from torch import nn

class encoder_decoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(encoder_decoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,vid_features,mode='train',targets=None):
        vid_features,hidden = self.encoder(vid_features)
        seq_probs = self.decoder(vid_features,hidden,mode,targets)
        return seq_probs