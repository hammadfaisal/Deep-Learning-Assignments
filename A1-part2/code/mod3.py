import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import pad
import sys
from transformers import BertTokenizerFast, BertModel


vec_dim = 100
tgt_max_pad = 32
src_max_pad = 64

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x, att_mask):
        return self.bert(x, att_mask)
    
# works only for num_layers = 1
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, tgt_embed):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_size+hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, tgt_embed.num_embeddings)
        self.tgt_embed = tgt_embed
        self.output_size = output_size
        self.att_fc = nn.Linear(output_size+hidden_size, hidden_size//4)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.att_weight = nn.Linear(hidden_size//4, 1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_normal_(self.att_fc.weight)
        nn.init.zeros_(self.att_fc.bias)

    def forward(self, enc_o, enc_c, enc_h, x_tgt = None, teacher_prob = 1,  beam_size = 1) :
        if self.training:
            teacher_forcing = (np.random.rand() < teacher_prob)

            x = x_tgt[:,0].unsqueeze(1)
            dec_out = torch.zeros(x.size(0), tgt_max_pad-1, self.tgt_embed.num_embeddings, device=enc_o.device)
            h_prev = enc_h
            c_prev = enc_c
            h_prev = h_prev.unsqueeze(0)
            c_prev = c_prev.unsqueeze(0)
            for i in range(tgt_max_pad-1):
                x = self.tgt_embed(x)
                att_act = self.att_fc(torch.cat([x.repeat((1,enc_o.size(1),1)), enc_o], dim=-1))
                att_act = self.att_weight(self.tanh(self.softmax(att_act)))
                context_vec = torch.sum(att_act*enc_o, dim=1).unsqueeze(1)
                
                x = torch.cat([x, context_vec], dim=-1)

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
            att_act = self.att_fc(torch.cat([x.repeat((1,enc_o.size(1),1)), enc_o], dim=-1))
            att_act = self.att_weight(self.tanh(self.softmax(att_act)))
            context_vec = torch.sum(att_act*enc_o, dim=1).unsqueeze(1)
            x = torch.cat([x, context_vec], dim=-1)
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
            enc_o = enc_o.repeat(1,beam_size,1).view(batch_size*beam_size,-1,enc_o.size(-1))

            choose_from = torch.arange(self.tgt_embed.num_embeddings, device=enc_c.device).view(1,1,-1).repeat(batch_size,beam_size,1).view(batch_size,-1)
            prev_from = torch.arange(beam_size, device=enc_c.device).unsqueeze(1).repeat(1, self.tgt_embed.num_embeddings).unsqueeze(0).repeat(batch_size,1,1).view(batch_size,-1)


            for i in range(1,tgt_max_pad-1):
                x = self.tgt_embed(x)

                att_act = self.att_fc(torch.cat([x.repeat((1,enc_o.size(1),1)), enc_o], dim=-1))
                att_act = self.att_weight(self.tanh(self.softmax(att_act)))
                context_vec = torch.sum(att_act*enc_o, dim=1).unsqueeze(1)
                x = torch.cat([x, context_vec], dim=-1)
                
                x, (h_prev, c_prev) = self.lstm(x, (h_prev.unsqueeze(0), c_prev.unsqueeze(0)))
                h_prev = h_prev.squeeze(0)
                c_prev = c_prev.squeeze(0)
                x = self.fc(x)
                # add score and take top k
                # change view to (batch_size, beam_size, -1)
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
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_src, att_mask, x_tgt = None, t_prob = 0.6, beam_size = 1):
        enc_out = self.encoder(x_src, att_mask)
        f_hidden = enc_out['last_hidden_state']
        cls_tok = enc_out['pooler_output']
    
        return self.decoder(f_hidden, cls_tok, cls_tok, x_tgt = x_tgt, teacher_prob = t_prob, beam_size = beam_size)
    
if __name__ == '__main__':

    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    loaded = torch.load(sys.argv[1], map_location=device)
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


    src_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def collate_batch(batch):
        src = [d['Problem'] for d in batch]
        tgt = []
        for d in batch:
            cur_sample = []
            for tok in d['linear_formula']:
                if tok in tgt_vocab.keys() :
                    cur_sample.append(tgt_vocab[tok])
                else:
                    cur_sample.append(tgt_vocab['<unk>'])
            tgt.append(cur_sample)
        src = src_tokenizer(src, truncation=True, return_tensors='pt', padding = 'max_length', max_length = src_max_pad)

        sos_idx = torch.tensor([tgt_vocab['<sos>']])
        eos_idx = torch.tensor([tgt_vocab['<eos>']])
        proc_tgt = []
        for tgt_sample in tgt:
            proc_tgt_sample = torch.cat([sos_idx, torch.tensor(tgt_sample), eos_idx])
            proc_tgt_sample = proc_tgt_sample[:tgt_max_pad]
            proc_tgt.append(pad(proc_tgt_sample, (0, tgt_max_pad-len(proc_tgt_sample)), value = tgt_vocab['<pad>']))
        tgt = torch.stack(proc_tgt)
        return src['input_ids'], src['attention_mask'], tgt


    def train(model, criterion, optimizer, train_loader, device , batch_size):
        model.train()
        total_loss = 0
        exact_match = 0
        tot_samples = 0
        for i, (src_input_id, src_att_mask, tgt) in enumerate(train_loader):
            model = model.to(device)
            src_input_id = src_input_id.to(device)
            src_att_mask = src_att_mask.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            out = model(src_input_id, src_att_mask, tgt[:,:-1])
            loss = criterion(out.permute(0,2,1), tgt[:,1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
            optimizer.step()
            total_loss += loss.item()*src_input_id.size(0)
            pred = out.argmax(dim=-1)
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
            tot_samples += src_input_id.size(0)
        return total_loss/tot_samples, exact_match/(tot_samples)

    def validate(model, criterion, val_loader, device, batch_size):
        model.eval()
        total_loss = 0
        exact_match = 0
        tot_samples = 0
        with torch.no_grad():
            for i, (src_input_id, src_att_mask, tgt) in enumerate(val_loader):
                model = model.to(device)
                src_input_id = src_input_id.to(device)
                src_att_mask = src_att_mask.to(device)
                tgt = tgt.to(device)
                out = model(src_input_id, src_att_mask,beam_size = 1)
                loss = criterion(out.permute(0,2,1), tgt[:,1:])
                total_loss += loss.item()*src_input_id.size(0)
                pred = out.argmax(dim=-1)
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
                tot_samples += src_input_id.size(0)
        return total_loss/tot_samples, exact_match/(tot_samples)


    val_data = None
    with open('data/dev.json') as f:
        val_data = json.load(f)

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


    batch_size = 32
    weight_decay = 0.00001

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    encoder = Encoder()
    decoder = Decoder(vec_dim, 768, 1, tgt_embedding)
    model = Seq2Seq(encoder, decoder)
    model = model.to(device)
    model.encoder.freeze_bert()

    criterion = nn.CrossEntropyLoss(ignore_index = tgt_vocab['<pad>'])


    optimizer = optim.Adam(model.parameters(), lr = 0.00005, weight_decay = weight_decay)
    

    epochs = 200

    train_stats = []
    val_stats = []

    val_best = 0

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
            torch.save({'model_state_dict':model.state_dict(),'tgt_vocab':tgt_vocab ,
                        'args_tok':tgt_args_tok, 'ops_tok':tgt_ops_tok}, 'model_best.pth')
        else:
            bad_epochs += 1

        if bad_epochs >=15 and epoch > 30:
            break


    np.save('train_stats.npy', np.array(train_stats))
    np.save('val_stats.npy', np.array(val_stats))


