import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import pad
import sys
import os


from transformers import BertTokenizerFast, BertModel

vec_dim = 100
tgt_max_pad = 32
src_max_pad = 64

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')  

def eval_bert(checkpoint, test_file, beam_size, out_file):

    from code.mod3 import Encoder, Decoder, Seq2Seq

    data = None

    with open(test_file) as f:
        data = json.load(f)


    src_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    loaded = torch.load(checkpoint, map_location=device)

    tgt_vocab = loaded['tgt_vocab']

    tgt_embedding = nn.Embedding(len(tgt_vocab), vec_dim, padding_idx = tgt_vocab['<pad>'])

    def collate_batch_test(batch):
        src = [d['Problem'] for d in batch]
        src = src_tokenizer(src, truncation=True, return_tensors='pt', padding = 'max_length', max_length = src_max_pad)
        return src['input_ids'], src['attention_mask']


    encoder = Encoder()
    decoder = Decoder(vec_dim, 768, 1, tgt_embedding)
    model = Seq2Seq(encoder, decoder)
    model = model.to(device)
    model.load_state_dict(loaded['model_state_dict'])

    model.eval()

    test_loader = DataLoader(data, batch_size=1, collate_fn=collate_batch_test)

    out_data = []
    inv_tgt_vocab = {v:k for k,v in tgt_vocab.items()}

    for src, mask in test_loader:
        src = src.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            out = model(src, mask, beam_size=beam_size)
            out = out.argmax(dim=-1)
            out_data.append(out)


    out_data = [d.cpu().numpy().tolist() for d in out_data]

    args = loaded['args_tok']
    ops = loaded['ops_tok']


    with open(test_file) as f:
        data = json.load(f)
    
    for i in range(len(data)):
        unproc = out_data[i]
        ans = ""
        # 0,1,2 . 0-> Nothing, 1-> Op, 2->Arg
        prev = 0
        for tok in unproc[0]:
            if tok == tgt_vocab['<eos>']:
                break
            word = inv_tgt_vocab[tok]
            if (word in ops):
                if prev == 2:
                    ans += ')'
                if prev != 0:
                    ans += '|'
                prev = 1
            elif (word in args) :
                if prev == 1:
                    ans += '('
                if prev == 2:
                    ans += ','
                prev = 2
            else :
                continue
            ans += word
        if (prev == 2):
            ans += ')'
        data[i]['predicted'] = ans


    with open(out_file, 'w') as f:
        json.dump(data, f, indent=4)




def eval_non_bert(checkpoint, test_path, model_type, beam_size, out_file):

    if model_type == 'attn':
        from code.mod2 import Seq2Seq
    else:
        from code.mod1 import Seq2Seq

    loaded = torch.load(checkpoint, map_location=device)

    tgt_vocab = loaded['tgt_vocab']
    src_vocab = loaded['src_vocab']
    args = loaded['args_tok']
    ops = loaded['ops_tok']

    tgt_embedding = nn.Embedding(len(tgt_vocab), vec_dim, padding_idx = tgt_vocab['<pad>'])
    src_embedding = nn.Embedding(len(src_vocab), vec_dim, padding_idx = src_vocab['<pad>'])

    def collate_batch_test(batch, device, src_vocab) :
            src_list = []
            for src in batch:
                proc_src = torch.tensor(src[:src_max_pad])
                src_list.append(pad(proc_src, (0, src_max_pad-len(proc_src)), value = src_vocab['<pad>']))

            src_list = torch.stack(src_list).to(device)

            return src_list


    model = Seq2Seq(vec_dim, 200, 1, src_embedding, tgt_embedding, vec_dim)
    model = model.to(device)
    model.load_state_dict(loaded['model_state_dict'])

    model.eval()

    data = None

    with open(test_path) as f:
        data = json.load(f)

    fdata = []

    for d in data:
            d['Problem'] = d['Problem'].split()

            src = []
            for tok in d['Problem']:
                if tok in src_vocab:
                    src.append(src_vocab[tok])
                else :
                    src.append(src_vocab['<unk>'])
            fdata.append(src)

    test_loader = DataLoader(fdata, batch_size=1, collate_fn=lambda x: collate_batch_test(x, device, src_vocab))

    out_data = []

    for src in test_loader:
        src = src.to(device)
        with torch.no_grad():
            out = model(src, beam_size=beam_size)
            out = out.argmax(dim=-1)
            out_data.append(out)

    out_data = [d.cpu().numpy().tolist() for d in out_data]

    inv_tgt_vocab = {v:k for k,v in tgt_vocab.items()}
    args = loaded['args_tok']
    ops = loaded['ops_tok']


    with open(test_path) as f:
        data = json.load(f)

    for i in range(len(data)):
        unproc = out_data[i]
        ans = ""
        # 0,1,2 . 0-> Nothing, 1-> Op, 2->Arg
        prev = 0
        for tok in unproc[0]:
            if tok == tgt_vocab['<eos>']:
                break
            word = inv_tgt_vocab[tok]
            if (word in ops):
                if prev == 2:
                    ans += ')'
                if prev != 0:
                    ans += '|'
                prev = 1
            elif (word in args) :
                if prev == 1:
                    ans += '('
                if prev == 2:
                    ans += ','
                prev = 2
            else :
                continue
            ans += word
        if (prev == 2):
            ans += ')'
        data[i]['predicted'] = ans


    with open(out_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    if (len(sys.argv) != 9):
        print("Usage: python3 infer.py --model_file <path to the trained model> --beam_size [1 | 10 | 20 ]  --model_type [ lstm_lstm | lstm_lstm_attn | bert_lstm_attn_frozen | bert_lstm_attn_tuned] --test_data_file <the json file containing the problems>")
        sys.exit(1)
    
    model_file = sys.argv[2]
    beam_size = int(sys.argv[4])
    model_type = sys.argv[6]
    test_data_file = sys.argv[8]

    if model_type == 'lstm_lstm' or model_type == 'lstm_lstm_attn':
        eval_non_bert(model_file, test_data_file, model_type[-4:], beam_size, test_data_file)
    else:
        eval_bert(model_file, test_data_file, beam_size, test_data_file)
