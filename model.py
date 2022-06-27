import numpy as np
import torch
import dataloader

config = {}

# config['batch_size'] = 4096
config['bpr_batch_size'] = int(2048)
config['latent_dim_rec'] = int(64)
config['lightGCN_n_layers'] = int(3)
config['dropout'] = int(0)
config['keep_prob'] = float(0.6)
config['A_n_fold'] = int(100)
config['test_u_batch_size'] = int(100)
config['multicore'] = int(0)
config['lr'] = float(0.001)
config['decay'] = float(1e-4)
config['pretrain'] = int(0)
config['A_split'] = False
config['bigdata'] = False




"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
# import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError



class LightGCN(BasicModel):
    def __init__(self,
                 config:dict,
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            # world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN





        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # print(g_droped,all_emb)
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


import dataloader
dataset = dataloader.Loader(path="taobao/")
dataset.getSparseGraph()

# print(dataset.n_users)
# print(dataset.m_items)

Recmodel = LightGCN(config, dataset)



# item_num = dataset.m_items




































'''
-------------------------------------------------------


以上为LightGCN部分


------------------------------------------------------
'''



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class AttentionLayer(torch.nn.Module):
    """A pytorch implementation of Reference:"""
    def __init__(self, embed_dim, attn_size, dropout):
        super().__init__()
        self.fc = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        """
        param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        attn_scores = torch.relu(self.fc(x))
        gate_scores = torch.softmax(self.projection(attn_scores), dim=1)
        gate_output = torch.sum(gate_scores * x, dim=1)
        gate_output = self.dropout(gate_output)

        # print(gate_output)
        # print('------------------------------------------')
        # print(gate_output.shape)

        return gate_output


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        # self.item_emb = torch.nn.Embedding(236033, args.hidden_units, padding_idx=0)
        # self.item_emb = torch.nn.Embedding(115224,     # jd
        #                                    args.hidden_units,
        #                                    padding_idx=0)
        self.item_emb = torch.nn.Embedding(228038,    #taobao
                                           args.hidden_units,
                                           padding_idx=0)

        # self.item_emb.weight.data[1:].copy_(torch.cat((Recmodel.computer()[0],Recmodel.computer()[1]),0)) # attr
        self.item_emb.weight.data[1:].copy_(Recmodel.computer()[1])
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

        self.attn1 = AttentionLayer(args.hidden_units,
                                        attn_size=32,
                                        dropout=0.2)
        self.attn2 = AttentionLayer(args.hidden_units,
                                    attn_size=32,
                                    dropout=0.2)
        self.attn3 = AttentionLayer(args.hidden_units,
                                    attn_size=32,
                                    dropout=0.2)
        self.attn4 = AttentionLayer(args.hidden_units,
                                    attn_size=32,
                                    dropout=0.2)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])  # 沿y轴复制log_seqs.shape[0]
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # sum_attention = torch.zeros(128,200,200)
        # ff = open('attention_weight.txt','w')

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q  = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)

            # torch.set_printoptions(threshold=np.inf)
            # sum_attention.add_(_)
            # print(sum_attention,file=ff)

            # print(self.attention_layers)

            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            # print('---------------------')
            # print(seqs)
            # print(seqs.shape())
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)



        return log_feats




    # def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
    #     log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

    #     pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
    #     neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

    #     pos_logits = (log_feats * pos_embs).sum(dim=-1)
    #     neg_logits = (log_feats * neg_embs).sum(dim=-1)

    #     # pos_pred = self.pos_sigmoid(pos_logits)
    #     # neg_pred = self.neg_sigmoid(neg_logits)

    #     return pos_logits, neg_logits # pos_pred, neg_pred







    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, user_ids1,
                log_seqs1, pos_seqs1, neg_seqs1, user_ids2, log_seqs2,
                pos_seqs2, neg_seqs2, user_ids3, log_seqs3, pos_seqs3,
                neg_seqs3):  # for training
        log_feats0 = self.log2feats(log_seqs)  # user_ids hasn't been used yet
        log_feats0 = torch.unsqueeze(log_feats0, 1)
        log_feats1 = self.log2feats(log_seqs1)  # user_ids hasn't been used yet
        log_feats1 = torch.unsqueeze(log_feats1, 1)
        log_feats2 = self.log2feats(log_seqs2)  # user_ids hasn't been used yet
        log_feats2 = torch.unsqueeze(log_feats2, 1)
        log_feats3 = self.log2feats(log_seqs3)  # user_ids hasn't been used yet
        log_feats3 = torch.unsqueeze(log_feats3, 1)
        log_feats = torch.cat((log_feats0, log_feats1, log_feats2, log_feats3),
                              1)
        log_featsa = self.attn1(log_feats)

        log_featsb = self.attn2(log_feats)
        log_featsc = self.attn3(log_feats)
        log_featsd = self.attn4(log_feats)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_featsa * pos_embs).sum(dim=-1)
        neg_logits = (log_featsa * neg_embs).sum(dim=-1)

        pos_embs1 = self.item_emb(torch.LongTensor(pos_seqs1).to(self.dev))
        neg_embs1 = self.item_emb(torch.LongTensor(neg_seqs1).to(self.dev))
        pos_logits1 = (log_featsb * pos_embs1).sum(dim=-1)
        neg_logits1 = (log_featsb * neg_embs1).sum(dim=-1)

        pos_embs2 = self.item_emb(torch.LongTensor(pos_seqs2).to(self.dev))
        neg_embs2 = self.item_emb(torch.LongTensor(neg_seqs2).to(self.dev))
        pos_logits2 = (log_featsc * pos_embs2).sum(dim=-1)
        neg_logits2 = (log_featsc * neg_embs2).sum(dim=-1)

        pos_embs3 = self.item_emb(torch.LongTensor(pos_seqs3).to(self.dev))
        neg_embs3 = self.item_emb(torch.LongTensor(neg_seqs3).to(self.dev))
        pos_logits3 = (log_featsd * pos_embs3).sum(dim=-1)
        neg_logits3 = (log_featsd * neg_embs3).sum(dim=-1)






        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits, pos_logits1, neg_logits1, pos_logits2, neg_logits2, pos_logits3, neg_logits3  # pos_pred, neg_pred


    # def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, user_ids1,
    #             log_seqs1, pos_seqs1, neg_seqs1):  # for training
    #     log_feats0 = self.log2feats(log_seqs)  # user_ids hasn't been used yet
    #     log_feats1 = self.log2feats(log_seqs1)  # user_ids hasn't been used yet
    #     log_feats = torch.cat((log_feats0, log_feats1), 1)

    #     pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
    #     neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
    #     pos_embs = torch.cat((pos_embs, pos_embs), 1)
    #     neg_embs = torch.cat((neg_embs, neg_embs), 1)

    #     pos_logits = (log_feats * pos_embs).sum(dim=-1)
    #     neg_logits = (log_feats * neg_embs).sum(dim=-1)

    #     pos_embs1 = self.item_emb(torch.LongTensor(pos_seqs1).to(self.dev))
    #     neg_embs1 = self.item_emb(torch.LongTensor(neg_seqs1).to(self.dev))
    #     pos_embs1 = torch.cat((pos_embs1, pos_embs1), 1)
    #     neg_embs1 = torch.cat((neg_embs1, neg_embs1), 1)

    #     pos_logits1 = (log_feats * pos_embs1).sum(dim=-1)
    #     neg_logits1 = (log_feats * neg_embs1).sum(dim=-1)

    #     # pos_pred = self.pos_sigmoid(pos_logits)
    #     # neg_pred = self.neg_sigmoid(neg_logits)

    #     return pos_logits, neg_logits, pos_logits1, neg_logits1  # pos_pred, neg_pred



    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
