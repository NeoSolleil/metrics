#coding:utf-8
'''
6.3.5.4 Transformer - PyTorch
'''

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from layers.torch import PositionalEncoding
from layers.torch import MultiHeadAttention
from utils import Vocab
import pickle
import nltk
import re

import itertools

import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class Transformer(nn.Module):
    def __init__(self,
                 depth_source,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(depth_source,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               maxlen=maxlen,
                               device=device)
        self.decoder = Decoder(depth_target,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               maxlen=maxlen,
                               device=device)
        self.out = nn.Linear(d_model, depth_target)
        nn.init.xavier_normal_(self.out.weight)

        self.maxlen = maxlen

    def forward(self, source, target=None):

        mask_source = self.sequence_mask(source)
        hs = self.encoder(source, mask=mask_source)

        return hs

        if target is not None:
            target = target[:, :-1]
            len_target_sequences = target.size(1)
            mask_target = self.sequence_mask(target).unsqueeze(1)
            subsequent_mask = self.subsequence_mask(target)
            mask_target = torch.gt(mask_target + subsequent_mask, 0)

            y = self.decoder(target, hs,
                             mask=mask_target,
                             mask_source=mask_source)
            output = self.out(y)
        else:
            batch_size = source.size(0)
            len_target_sequences = self.maxlen

            output = torch.ones((batch_size, 1),
                                dtype=torch.long,
                                device=self.device)

            for t in range(len_target_sequences - 1):
                mask_target = self.subsequence_mask(output)
                out = self.decoder(output, hs,
                                   mask=mask_target,
                                   mask_source=mask_source)
                out = self.out(out)[:, -1:, :]
                out = out.max(-1)[1]
                output = torch.cat((output, out), dim=1)

        return output

    def sequence_mask(self, x):
        return x.eq(0)

    def subsequence_mask(self, x):
        shape = (x.size(1), x.size(1))
        mask = torch.triu(torch.ones(shape, dtype=torch.bool),
                          diagonal=1)
        return mask.unsqueeze(0).repeat(x.size(0), 1, 1).to(self.device)
class Encoder(nn.Module):
    def __init__(self,
                 depth_source,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(depth_source,
                                      d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(h=h,
                         d_model=d_model,
                         d_ff=d_ff,
                         p_dropout=p_dropout,
                         maxlen=maxlen,
                         device=device) for _ in range(N)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        y = self.pe(x)
        for encoder_layer in self.encoder_layers:
            y = encoder_layer(y, mask=mask)

        return y
class EncoderLayer(nn.Module):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128,
                 device='cpu'):
        super().__init__()
        self.attn = MultiHeadAttention(h, d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FFN(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        h = self.attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)
        y = self.ff(h)
        y = self.dropout2(y)
        y = self.norm2(h + y)

        return y
class Decoder(nn.Module):
    def __init__(self,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(depth_target,
                                      d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model, maxlen=maxlen)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(h=h,
                         d_model=d_model,
                         d_ff=d_ff,
                         p_dropout=p_dropout,
                         maxlen=maxlen,
                         device=device) for _ in range(N)
        ])

    def forward(self, x, hs,
                mask=None,
                mask_source=None):
        x = self.embedding(x)
        y = self.pe(x)

        for decoder_layer in self.decoder_layers:
            y = decoder_layer(y, hs,
                              mask=mask,
                              mask_source=mask_source)
        return y
class DecoderLayer(nn.Module):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 maxlen=128,
                 device='cpu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.src_tgt_attn = MultiHeadAttention(h, d_model)
        self.dropout2 = nn.Dropout(p_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FFN(d_model, d_ff)
        self.dropout3 = nn.Dropout(p_dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, hs,
                mask=None,
                mask_source=None):
        h = self.self_attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)

        z = self.src_tgt_attn(h, hs, hs,
                              mask=mask_source)
        z = self.dropout2(z)
        z = self.norm2(h + z)

        y = self.ff(z)
        y = self.dropout3(y)
        y = self.norm3(z + y)

        return y
class FFN(nn.Module):
    def __init__(self, d_model, d_ff,
                 device='cpu'):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        h = self.l1(x)
        h = self.a1(h)
        y = self.l2(h)
        return y

if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    1. データの準備
    '''
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    MT = os.path.join(data_dir, '/MT_PATH')#MT訳へのパスを設定
    ja = os.path.join(data_dir, '/REF_PATH')#参照訳へのパスを設定



    en_vocab = Vocab()
    ja_vocab = Vocab()

    with open ('/en.pickle', mode='rb') as f:#deeplearning_torch/08_transformer_torch.pyで生成した英語のボキャブラリーへのパスの設定
        en_vocab.w2i = pickle.load(f)
    en_vocab.i2w = {i: w for w, i in en_vocab.w2i.items()}

    with open ('/ja.pickle', mode='rb') as f:#deeplearning_torch/08_transformer_torch.pyで生成した日本語のボキャブラリーへのパスの設定
        ja_vocab.w2i = pickle.load(f)
    ja_vocab.i2w = {i: w for w, i in ja_vocab.w2i.items()}



    def sort(x, t):
        lens = [len(i) for i in x]
        indices = sorted(range(len(lens)), key=lambda i: -lens[i])
        x = [x[i] for i in indices]
        t = [t[i] for i in indices]

        return (x, t)


    '''
    2. モデルの構築
    '''
    depth_x = len(en_vocab.i2w)
    depth_t = len(ja_vocab.i2w)
    model = Transformer(depth_x,
                        depth_t,
                        N=3,
                        h=4,
			            d_model=128,
                        d_ff=256,
                        #maxlen=20,
                        maxlen=240,
                        device=device).to(device)



    load_model=model.load_state_dict(torch.load('/model_PATH'))#deeplearning_torch/08_transformer_torch.pyで生成したモデルへのパスを設定


    '''
    3. モデルの学習・評価
    '''

    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.0001,
                                betas=(0.9, 0.999), amsgrad=True)

    def compute_loss(label, pred):
        return criterion(pred, label)

    def train_step(x, t):
        model.train()
        preds = model(x, t)
        loss = compute_loss(t[:, 1:].contiguous().view(-1),
                            preds.contiguous().view(-1, preds.size(-1)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    def val_step(x, t):
        model.eval()
        preds = model(x, t)
        loss = compute_loss(t[:, 1:].contiguous().view(-1),
                            preds.contiguous().view(-1, preds.size(-1)))

        return loss, preds

    def test_step(x):
        model.eval()
        preds = model(x)
        return preds

    BLEU_out = []
    BLEU_target = []
    B_target = ""
    B_out = ""
    target_sum = 0
    source_sum = 0
    targetU3_num = 0
    sourceU3_num = 0
    target4to6_num = 0
    source4to6_num = 0
    targetO7_num = 0
    sourceO7_num = 0

    f1 = open(MT,'r')
    MT = f1.readlines()
    f1.close()
    f2 = open(ja,'r')
    ja = f2.readlines()
    f2.close()

    for k in range(len(ja)):#len(ai)
        i = MT[k]#MT訳
        j = ja[k]#参照訳


        sannsyou = j.split()

        x_test = en_vocab.encode(sannsyou)
        x = x_test
        x = [x]
        x = torch.tensor(x)
        sannsyou_preds = test_step(x)

        aq=sannsyou_preds.tolist()

        aaq=list(itertools.chain.from_iterable(aq))

        context_ref = np.zeros(shape=(128))
        for one_wd_emb in aaq:

            context_ref = np.add(context_ref, one_wd_emb)

        mtyaku = i.split()

        y_test = en_vocab.encode(mtyaku)
        y = y_test
        y = [y]
        y = torch.tensor(y)

        mtyaku_preds = test_step(y)

        az=mtyaku_preds.tolist()

        aaz=list(itertools.chain.from_iterable(az))

        context_MT = np.zeros(shape=(128))
        for one_wd_emb in aaz:

            context_MT = np.add(context_MT, one_wd_emb)


        z=cos_sim(context_ref, context_MT)
        print(z)






