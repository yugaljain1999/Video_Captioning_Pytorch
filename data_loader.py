import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self,mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        self.captions = json.load(open("captions.json"))
        info = json.load(open("info.json")) # info of splits into training and testing dataset,vocab-(idx to word and word to idx)
        self.ix_to_word = info['idx_to_word']
        self.word_to_ix = info['word_to_idx']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['validate']))
        print('number of test videos: ', len(self.splits['test']))

        self.feats_dir = ['data/feats/inception_v3'] # video features saved in numpy binary format
        #self.c3d_feats_dir = opt['c3d_feats_dir']
        #self.with_c3d = opt['with_c3d']
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = 50
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        # which part of data to load
        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['validate'])
        
        fc_feat = []
        for dir in self.feats_dir:
            if not os.path.exists(os.path.join(dir, 'video%i.npy' % (ix))):
                continue
            fc_feat.append(np.load(os.path.join(dir, 'video%i.npy' % (ix))))
        #print('fc_feat',fc_feat)
        fc_feat = np.concatenate(fc_feat, axis=1)
        #if self.with_c3d == 1:
        #    c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy'%(ix)))
        #    c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
        #    fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
        label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        captions = self.captions['video%i'%(ix)]['final_captions']
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<EOS>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w] # for each caption pass id of each word to an array of shape(len(captions),maximum length))

        # random select a caption for this video
        cap_ix = random.randint(0, len(captions) - 1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0]) + 1] = 1 # masking for which label == 0

        data = {}
        data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        #data['gts'] = torch.from_numpy(gts).long()
        if os.path.exists(os.path.join(dir, 'video%i.npy' % (ix))):
            data['video_ids'] = 'video%i'%(ix)
        #print('fc_feats_size',data['fc_feats'].size())
        #print('labels_size',data['labels'].size())
        #print('masks_size',data['masks'].size())
        #print('gts_size',data['gts'].size())
        return data

    def __len__(self):
        return len(self.splits[self.mode])