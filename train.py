import json
import os

import numpy as np

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from data_loader import VideoDataset
#from misc.rewards import get_self_critical_reward, init_cider_scorer
from model import DecoderRNN, EncoderRNN
from S2VTAttModel import encoder_decoder
from torch import nn
from torch.utils.data import DataLoader
#from pretrainedmodels import utils
import torch
import torch.nn as nn
from torch.autograd import Variable


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


class RewardCriterion(nn.Module):

    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.contiguous().view(-1)
        reward = reward.contiguous().view(-1)
        mask = (seq > 0).float()
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1).cuda(),
                         mask[:, :-1]], 1).contiguous().view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output
def train(loader, model, optimizer, lr_scheduler):
    # Training
    criteria = nn.CrossEntropyLoss()
    model.train()
    #model = nn.DataParallel(model)
    for epoch in range(100):
       

        iteration = 0
        # If start self crit training
        #if -1 != -1 and epoch >= -1:
        #    sc_flag = True
        #    init_cider_scorer('msr-all-idxs')
        #else:
        #    sc_flag = False
        
        for data in loader:
            #print(data)
            torch.cuda.synchronize()
            fc_feats = data['fc_feats'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()

            optimizer.zero_grad()
            #if not sc_flag:
            seq_probs = model(fc_feats,mode = 'train',targets=labels)
            
            #loss = criteria(seq_probs,labels[:,0:])
            crit = LanguageModelCriterion()
            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            #else:
            #seq_probs, seq_preds = model(
            #        fc_feats, mode='inference', opt=opt)
            #reward = get_self_critical_reward(model, fc_feats, data,
            #                                      seq_preds)
            #print(reward.shape)
            #loss = rl_crit(seq_probs, seq_preds,
            #                   torch.from_numpy(reward).float().cuda())
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
           
            clip_grad_value_(model.parameters(), 5)
            optimizer.step()
            lr_scheduler.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1
            if iteration==38:
                break
            print("iter %d (epoch %d), train_loss = %.6f" %
                      (iteration, epoch, train_loss))
            #else:
            #    print("iter %d (epoch %d), avg_reward = %.6f" %
            #          (iteration, epoch, np.mean(reward[:, 0])))

        if epoch % 2 == 0:
            model_path = os.path.join('checkpoints',
                                      'model_%d.pth' % (epoch))
            model_info_path = os.path.join('checkpoints',
                                           'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))


def main():
    dataset = VideoDataset('train')
    dataloader = DataLoader(dataset, batch_size=128)
    
    #opt["vocab_size"] = dataset.get_vocab_size()
    #if opt["model"] == 'S2VTModel':
    #    model = S2VTModel(
    #        dataset.get_vocab_size(),
    #        50,
    #        512,
    #        256,
    #        224,
    #        rnn_cell='',
    #        n_layers=opt['num_layers'],
    #        rnn_dropout_p=opt["rnn_dropout_p"])
    #elif opt["model"] == "S2VTAttModel":
    encoder = EncoderRNN(
        2048,
        512,
        1,
        bidirectional=False)
        #input_dropout_p=opt["input_dro,
        #rnn_cell=opt['rnn_type'],
        #rnn_dropout_p=opt["rnn_dropout_p"])
    decoder = DecoderRNN(
        dataset.get_vocab_size(),
        512,
        28,
        512,
        1)
        #input_dropout_p=opt["input_dropout_p"],
        #rnn_cell=opt['rnn_type'],
        #rnn_dropout_p=opt["rnn_dropout_p"],
        #bidirectional=opt["bidirectional"])
    model = encoder_decoder(encoder, decoder)
    model = model.cuda()
    #crit = utils.LanguageModelCriterion()
    #rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=5e-4)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=200)

    train(dataloader, model, optimizer, exp_lr_scheduler)


if __name__ == '__main__':
    #opt = opts.parse_opt()
    #opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #opt_json = os.path.join('checkpoints', 'opt_info.json')
    #if not os.path.isdir(opt["checkpoint_path"]):
    #    os.mkdir(opt["checkpoint_path"])
    #with open(opt_json, 'w') as f:
    #    json.dump(opt, f)
    #print('save opt details to %s' % (opt_json))
    main()
