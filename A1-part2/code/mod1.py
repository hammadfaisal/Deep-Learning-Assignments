import re
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import pad


vec_dim = 100
src_max_pad = 64
tgt_max_pad = 32

class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, src_embed):
        super(Seq2SeqEncoder, self).__init__()
        # bi lstm
        self.src_embed = src_embed
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        
    def forward(self, x):
        x = self.src_embed(x)
        _, (h, c) = self.lstm(x)
        return h,c
        
    
class Seq2SeqDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, tgt_embed):
        super(Seq2SeqDecoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, tgt_embed.num_embeddings)
        self.tgt_embed = tgt_embed
        self.output_size = output_size

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, enc_c, enc_h, x_tgt = None, teacher_prob = 1,  beam_size = 1) :
        if self.training:
            teacher_forcing = (np.random.rand() < teacher_prob)

            x = x_tgt[:,0].unsqueeze(1)
            dec_out = torch.zeros(x.size(0), tgt_max_pad-1, self.tgt_embed.num_embeddings, device=enc_c.device)
            h_prev = enc_h
            c_prev = enc_c
            h_prev = h_prev.unsqueeze(0)
            c_prev = c_prev.unsqueeze(0)
            for i in range(tgt_max_pad-1):
                x = self.tgt_embed(x)

                x, (h_prev, c_prev) = self.lstm(x, (h_prev, c_prev))
                x = self.fc(x)
                dec_out[:,i] = x.squeeze(1)
                if i == tgt_max_pad-2:
                    break
                if teacher_forcing:
                    x = x_tgt[:,i+1].unsqueeze(1)
                else:
                    x = x.argmax(dim=-1)
                # if all eos or pad tokens break . eos = 2 , pad = 3
                if ((x == 2) | (x == 3)).all():
                    break
            return dec_out
        else:

            batch_size = enc_c.size(0)
            # sos token
            x = torch.ones(batch_size, device=enc_c.device,dtype=torch.long).unsqueeze(1)

            # 1st iter start
            x = self.tgt_embed(x)
            x, (h_prev, c_prev) = self.lstm(x, (enc_h.unsqueeze(0), enc_c.unsqueeze(0)))
            x = self.fc(x)
            x_temp = x
            x = torch.softmax(x, dim=-1)
            x = torch.log(x)
            # score init
            score, idx = x.squeeze(1).topk(beam_size, dim=-1)
            x = idx.view(batch_size*beam_size, -1)
            x_temp = x_temp.repeat(1,beam_size,1)
            # best sentence init
            best_sentence = torch.zeros(batch_size, beam_size, tgt_max_pad-1, self.tgt_embed.num_embeddings, device=enc_c.device)
            best_sentence[:,:,0] = x_temp
            # 1st iter complete

            h_prev = h_prev.squeeze(0).repeat(1,beam_size).view(batch_size*beam_size,-1)
            c_prev = c_prev.squeeze(0).repeat(1,beam_size).view(batch_size*beam_size,-1)

            choose_from = torch.arange(self.tgt_embed.num_embeddings, device=enc_c.device).view(1,1,-1).repeat(batch_size,beam_size,1).view(batch_size,-1)
            prev_from = torch.arange(beam_size, device=enc_c.device).unsqueeze(1).repeat(1, self.tgt_embed.num_embeddings).unsqueeze(0).repeat(batch_size,1,1).view(batch_size,-1)


            for i in range(1,tgt_max_pad-1):
                x = self.tgt_embed(x)
                x, (h_prev, c_prev) = self.lstm(x, (h_prev.unsqueeze(0), c_prev.unsqueeze(0)))
                h_prev = h_prev.squeeze(0)
                c_prev = c_prev.squeeze(0)
                x = self.fc(x)
                # add score and take top k
                x = x.view(batch_size, beam_size, -1)
                x_temp = x
                # do softmax
                x = torch.softmax(x, dim=-1)
                # take log and add to score
                x = torch.log(x)
                score = score.unsqueeze(2) + x
                # use top k to get next beam x, h_prev and c_prev
                score, idx = score.view(batch_size, -1).topk(beam_size, dim=-1)
                # get x for next iteration using idx and choose_from
                x = choose_from[torch.arange(batch_size).unsqueeze(1).repeat(1,beam_size), idx]
                prev_beam_idx = prev_from[torch.arange(batch_size).unsqueeze(1).repeat(1,beam_size), idx]
                best_sentence = best_sentence[torch.arange(batch_size).unsqueeze(-1),prev_beam_idx]
                best_sentence[:,:,i] = x_temp
                x = x.view(batch_size*beam_size,-1)
                # get h_prev and c_prev
                h_prev = h_prev.view(batch_size, beam_size, -1)
                c_prev = c_prev.view(batch_size, beam_size, -1)
                h_prev = h_prev[torch.arange(batch_size).unsqueeze(-1), prev_beam_idx]
                c_prev = c_prev[torch.arange(batch_size).unsqueeze(-1), prev_beam_idx]
                # shape back to (batch_size*beam_size, -1)
                h_prev = h_prev.view(-1, h_prev.size(-1))
                c_prev = c_prev.view(-1, c_prev.size(-1))
                # if all eos or pad tokens break . eos = 2 , pad = 3
                if ((x == 2) | (x == 3)).all():
                    break
            return best_sentence[:,0]
            
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, src_embed, tgt_embed , output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Seq2SeqEncoder(input_size, hidden_size, num_layers, src_embed)
        self.decoder = Seq2SeqDecoder(output_size, 2*hidden_size, num_layers, tgt_embed)
    
    def forward(self, x_src, x_tgt = None, t_prob = 0.6 , beam_size = 1):
        h, c = self.encoder(x_src)
        # change to batch first (batch_size, hidden_size) for uniformity in all model decoders
        h = h.permute(1,0,2).contiguous().view(h.size(1), -1)
        c = c.permute(1,0,2).contiguous().view(c.size(1), -1)
        return self.decoder(c, h, x_tgt, t_prob, beam_size)



if __name__ == '__main__':

    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')       

    data = None
    with open('data/train.json') as f:
        data = json.load(f)

    tgt_vocab_tok = set()
    tgt_args_tok = set()
    tgt_ops_tok = set()

    for d in data:
        prob_tok = []
        tok1 = d['linear_formula'].split('|')
        for tok in tok1:
            tok = tok.strip()
            if tok == '':
                continue
            cur = ""
            for i in range(len(tok)):
                if tok[i] == '(' or tok[i] == ')' or tok[i] == ',':
                    prob_tok.append(cur)
                    if tok[i] == '(':
                        tgt_ops_tok.add(cur)
                    else:
                        tgt_args_tok.add(cur)
                    cur = ""
                else:
                    cur += tok[i]
        
        for a in prob_tok:
            tgt_vocab_tok.add(a)
        
        d['linear_formula'] = prob_tok

    tgt_vocab_tok = list(tgt_vocab_tok)
    tgt_vocab_tok = ['<unk>', '<sos>', '<eos>', '<pad>'] + tgt_vocab_tok
    tgt_vocab = {tok: i for i, tok in enumerate(tgt_vocab_tok)}
    tgt_embedding = nn.Embedding(len(tgt_vocab), vec_dim, padding_idx = tgt_vocab['<pad>'])


    src_vocab_tok = {}
    for d in data:
        src_tok = d['Problem'].split()
        for a in src_tok:
            if a not in src_vocab_tok:
                src_vocab_tok[a] = 1
            else:
                src_vocab_tok[a] += 1
        d['Problem'] = src_tok

    min_freq = 0

    src_vocab_tok = {k: v for k, v in src_vocab_tok.items() if v >= min_freq}
    src_vocab_tok = ['<unk>', '<pad>'] + list(src_vocab_tok.keys())
    src_vocab = {tok: i for i, tok in enumerate(src_vocab_tok)}
    src_embedding = nn.Embedding(len(src_vocab), vec_dim, padding_idx = src_vocab['<pad>'])

    glove = torchtext.vocab.GloVe(name='6B', dim=vec_dim)

    for word in src_vocab.keys():
        if word in glove.stoi:
            src_embedding.weight.data[src_vocab[word]] = glove.vectors[glove.stoi[word]]


    fdata = []

    for d in data:
        src = []
        for tok in d['Problem']:
            if tok in src_vocab:
                src.append(src_vocab[tok])
            else :
                src.append(src_vocab['<unk>'])
        tgt = []
        for tok in d['linear_formula']:
            if tok in tgt_vocab:
                tgt.append(tgt_vocab[tok])
            else:
                tgt.append(tgt_vocab['<unk>'])
        fdata.append((src, tgt))


    def collate_batch(batch, device, src_vocab, tgt_vocab) :
        src_list = []
        tgt_list = []
        sos_idx = torch.tensor([tgt_vocab['<sos>']])
        eos_idx = torch.tensor([tgt_vocab['<eos>']])
        for (src, tgt) in batch:
            proc_tgt = torch.cat([sos_idx, torch.tensor(tgt), eos_idx])
            proc_tgt = proc_tgt[:tgt_max_pad]
            proc_src = torch.tensor(src[:src_max_pad])
            src_list.append(pad(proc_src, (0, src_max_pad-len(proc_src)), value = src_vocab['<pad>']))
            tgt_list.append(pad(proc_tgt, (0, tgt_max_pad-len(proc_tgt)), value = tgt_vocab['<pad>']))

        src_list = torch.stack(src_list).to(device)
        tgt_list = torch.stack(tgt_list).to(device)

        return src_list, tgt_list


    def train(model, criterion, optimizer, train_loader, device , batch_size):
        model.train()
        total_loss = 0
        exact_match = 0
        for i, (src, tgt) in enumerate(train_loader):
            model = model.to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1], t_prob = 0.6)
            output = output.permute(0,2,1)
            loss = criterion(output, tgt[:,1:])
            loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
            optimizer.step()
            output = output.permute(0,2,1)
            pred = output.argmax(dim=-1)
            for b in range(pred.size(0)):
                before_eos_match = True
                for j in range(pred.size(1)):
                    if tgt[b][j+1] != pred[b][j]:
                        before_eos_match = False
                        break
                    if pred[b][j] == tgt_vocab['<eos>']:
                        break
                if before_eos_match:
                    exact_match += 1
            total_loss += loss.item()
            # print(f"Batch: {i} Loss: {loss.item()}")
        return total_loss/(len(train_loader)), exact_match/(len(train_loader)*batch_size)

    def validate(model, criterion, val_loader, device , batch_size) :
        model.eval()
        total_loss = 0
        exact_match = 0
        with torch.no_grad() :
            for i, (src, tgt) in enumerate(val_loader):
                model = model.to(device)
                src = src.to(device)
                tgt = tgt.to(device)
                output = model(src)
                output = output.permute(0,2,1)
                loss = criterion(output, tgt[:,1:])
                output = output.permute(0,2,1)
                pred = output.argmax(dim=-1)
                for b in range(pred.size(0)):
                    before_eos_match = True
                    for j in range(pred.size(1)):
                        if tgt[b][j+1] != pred[b][j]:
                            before_eos_match = False
                            break
                        if pred[b][j] == tgt_vocab['<eos>']:
                            break
                    if before_eos_match:
                        exact_match += 1
                total_loss += loss.item()
            
        return total_loss/(len(val_loader)), exact_match/(len(val_loader)*batch_size)


    batch_size = 32
    weight_decay = 0.00001

    train_loader = DataLoader(fdata, batch_size=batch_size, shuffle=True, collate_fn = lambda x: collate_batch(x, device, src_vocab, tgt_vocab))


    model = Seq2Seq(vec_dim, 200, 1, src_embedding, tgt_embedding, vec_dim).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index = tgt_vocab['<pad>'])


    optimizer = optim.Adam(model.parameters(), lr = 0.0005, weight_decay = weight_decay)

    epochs = 125

    val_data = None
    with open('data/dev.json') as f:
        val_data = json.load(f)

    fval_data = []

    for d in val_data:
        prob_tok = []
        tok1 = d['linear_formula'].split('|')
        for tok in tok1:
            tok = tok.strip()
            if tok == '':
                continue
            cur = ""
            for i in range(len(tok)):
                if tok[i] == '(' or tok[i] == ')' or tok[i] == ',':
                    prob_tok.append(cur)
                    cur = ""
                else:
                    cur += tok[i]
        d['linear_formula'] = prob_tok
        d['Problem'] = d['Problem'].split()

        src = []
        for tok in d['Problem']:
            if tok in src_vocab:
                src.append(src_vocab[tok])
            else :
                src.append(src_vocab['<unk>'])
        tgt = []
        for tok in d['linear_formula']:
            if tok in tgt_vocab:
                tgt.append(tgt_vocab[tok])
            else:
                tgt.append(tgt_vocab['<unk>'])
        fval_data.append((src, tgt))

    val_loader = DataLoader(fval_data, batch_size=batch_size, shuffle=False, collate_fn = lambda x: collate_batch(x, device, src_vocab, tgt_vocab))

    val_best = 0
    print("Training Started")


    train_stats = []
    val_stats = []

    bad_epochs = 0


    for epoch in range(epochs):
        loss, acc = train(model, criterion, optimizer, train_loader, device, batch_size)
        val_loss, val_acc = validate(model, criterion, val_loader, device, batch_size)
        print(f'Epoch {epoch} : Train Loss = {loss} , Train Acc = {acc} , Val Loss = {val_loss} , Val Acc = {val_acc}')
        train_stats.append((loss, acc))
        val_stats.append((val_loss, val_acc))

        if val_acc > val_best:
            val_best = val_acc
            bad_epochs = 0
            torch.save({'model_state_dict':model.state_dict(),'tgt_vocab':tgt_vocab , 'src_vocab':src_vocab,
                        'args_tok':tgt_args_tok, 'ops_tok':tgt_ops_tok}, 'model_best.pth')
        else:
            bad_epochs += 1

        if bad_epochs >=15 and epoch > 30:
            break


    np.save('train_stats.npy', np.array(train_stats))
    np.save('val_stats.npy', np.array(val_stats))
        


