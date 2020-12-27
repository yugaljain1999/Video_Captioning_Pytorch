import torch
from torch import nn
import torchvision
import json
import re

def build_vocab(vid_dict,word_threshold):
    count = {}
    for vid,caps in vid_dict.items():
        for cap in caps['captions']:
            ws = re.sub(r'[.,!;?]',' ',cap).split()
            for w in ws:
                count[w] = count.get(w,0) + 1 # get count of word - dictionary of words with it's counts

    total_words = sum(count.values()) # total words in whole dataset
    # find number of bad words in a vocabulary less than word_threshold
    bad_words = [w for w,n in count.items() if n<=word_threshold]
    vocab = [w for w,n in count.items() if n>word_threshold]
    bad_count = sum(count[w] for w in bad_words) # count of bad words
    if bad_count > 0:
        vocab.append('<UNK>')
    for vid,caps in vid_dict.items():
        caps = caps['captions']
        vid_dict[vid]['final_captions'] = [] # initialize other list of captions with <SOS>,<EOS> and <UNK>
        for cap in caps:
            ws = re.sub(r'[.,!;?]',' ',cap).split()
            for w in ws:
                caption = ['<SOS>'] + [w if count.get(w,0)>word_threshold else '<UNK>' for w in ws] +['<EOS>']
                vid_dict[vid]['final_captions'].append(caption)

    return vocab

def main():
    videos = json.load(open('train_val_videodatainfo.json','r'))['sentences']
    video_caption = {}
    for i in videos:
        if i['video_id'] not in video_caption.keys():
            video_caption[i['video_id']] = {'captions':[]} # initialized key value pair for captions for each video id
        video_caption[i['video_id']]['captions'].append(i['caption'])
    vocab = build_vocab(video_caption,1)
    itow = {i+2:w for i,w in enumerate(vocab)}
    wtoi = {w:i+2 for i,w in enumerate(vocab)}
    wtoi['<SOS>'] =1
    itow[1] = '<SOS>'
    wtoi['<EOS>'] = 0
    itow[0] = '<EOS>'
    out = {}
    out['idx_to_word'] = itow
    out['word_to_idx'] = wtoi
    out['videos'] = {'train':[],'validate':[],'test':[]}
    videos = json.load(open('train_val_videodatainfo.json','r'))['videos']
    print(videos)
    for i in videos:
        out['videos'][i['split']].append(int(i['id']))
    # dump out and video_caption into json files
    json.dump(out,open('info.json','w'))
    json.dump(video_caption,open('captions.json','w'))

main()