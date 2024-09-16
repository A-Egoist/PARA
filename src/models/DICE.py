import torch
from torch import nn


class DICE(nn.Module):
    def __init__(self, backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config, graph=None):
        super().__init__()
        self.backbone = backbone  # 'MF' or 'LightGCN'
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.item_popularity = item_popularity  # tensor

        if self.backbone == 'MF':
            self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
            self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
            nn.init.normal_(self.user_embedding.weight, std=0.01)
            nn.init.normal_(self.item_embedding.weight, std=0.01)
        elif self.backbone == 'LightGCN':
            self.graph = graph
            self.user_embedding_0 = nn.Embedding(self.num_users, self.embedding_dim)  # the first layer
            self.item_embedding_0 = nn.Embedding(self.num_items, self.embedding_dim)  # the first layer
            self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)  # the mean of all layers
            self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)  # the mean of all layers
            nn.init.normal_(self.user_embedding_0.weight, std=0.01)
            nn.init.normal_(self.item_embedding_0.weight, std=0.01)

    def get_reg_loss(self, user_indices, item_i_indices, item_j_indices):
        if self.backbone == 'MF':
            user_vector = self.user_embedding.weight[user_indices]
            item_i_vector = self.item_embedding.weight[item_i_indices]
            item_j_vector = self.item_embedding.weight[item_j_indices]
        elif self.backbone == 'LightGCN':
            user_vector = self.user_embedding_0.weight[user_indices]
            item_i_vector = self.item_embedding_0.weight[item_i_indices]
            item_j_vector = self.item_embedding_0.weight[item_j_indices]
        reg_loss = (1 / 2) * (user_vector.norm(2).pow(2) + item_i_vector.norm(2).pow(2) + item_j_vector.norm(2).pow(2)) / float(len(user_indices))
        return reg_loss
    
    def get_embedding(self):
        if self.backbone == 'MF':
            user_embeddings, item_embeddings = self.user_embedding.weight, self.item_embedding.weight
            return user_embeddings, item_embeddings
        elif self.backbone == 'LightGCN':
            user_embeddings = self.user_embedding_0.weight
            item_embeddings = self.item_embedding_0.weight

            all_emb = torch.cat([user_embeddings, item_embeddings])
            all_emb = all_emb.to(self.device)
            embs = [all_emb]  # [all_emb_0, all_emb_1, all_emb_2, ..., all_emd_n]

            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(self.graph, all_emb)
                embs.append(all_emb)

            embs = torch.stack(embs, dim=1)
            final_embs = torch.mean(embs, dim=1)
            self.user_embedding.weight.data, self.item_embedding.weight.data = torch.split(final_embs, [self.num_users, self.num_items])
            return self.user_embedding.weight, self.item_embedding.weight
        
    def forward(self, user_indices, item_i_indices, item_j_indices):
        user_embeddings, item_embeddings = self.get_embedding()
        user_vector = user_embeddings[user_indices]
        item_i_vector = item_embeddings[item_i_indices]
        item_j_vector = item_embeddings[item_j_indices]
        reg_loss = self.get_reg_loss(user_indices, item_i_indices, item_j_indices)

        # TODO
        # item_popularity_array = self.item_popularity['popularity'].values
        item_popularity_np = self.item_popularity.cpu().numpy()
    
        DICE_size = self.embedding_dim // 2
        prediction_i = (user_vector * item_i_vector).sum(dim=-1)
        prediction_j = (user_vector * item_j_vector).sum(dim=-1)
        loss_click = -1 * ((prediction_i - prediction_j).sigmoid().log().sum())
        # dict_click = {'prediction_i': prediction_i, 
        #               'prediction_j': prediction_j}

        user_embedding_1 = self.user_embedding(user_indices.unique())[:, 0:DICE_size]
        item_i_embedding_1 = self.item_embedding(item_i_indices.unique())[:, 0:DICE_size]
        item_j_embedding_1 = self.item_embedding(item_j_indices.unique())[:, 0:DICE_size]
        user_embedding_2 = self.user_embedding(user_indices.unique())[:, DICE_size:]
        item_i_embedding_2 = self.item_embedding(item_i_indices.unique())[:, DICE_size:]
        item_j_embedding_2 = self.item_embedding(item_j_indices.unique())[:, DICE_size:]
        loss_discrepancy = -1 * ((user_embedding_1 - user_embedding_2).sum() + (item_i_embedding_1 - item_i_embedding_2).sum() + (item_j_embedding_1 - item_j_embedding_2).sum())
        # dict_discrepancy = {'user_embedding_1': user_embedding_1, 
        #                     'user_embedding_2': user_embedding_2, 
        #                     'item_i_embedding_1': item_i_embedding_1, 
        #                     'item_i_embedding_2': item_i_embedding_2, 
        #                     'item_j_embedding_1': item_j_embedding_1, 
        #                     'item_j_embedding_2': item_j_embedding_2}

        item_i_np = item_i_indices.cpu().numpy().astype(int)
        item_j_np = item_j_indices.cpu().numpy().astype(int)
        # TODO
        # pop_relation = item_popularity_np[item_i_np - 1] > item_popularity_np[item_j_np - 1]
        pop_relation = item_popularity_np[item_i_np] > item_popularity_np[item_j_np]
        user_O1 = user_indices[pop_relation]
        user_O2 = user_indices[~pop_relation]
        item_i_O1 = item_i_indices[pop_relation]
        item_j_O1 = item_j_indices[pop_relation]
        item_i_O2 = item_i_indices[~pop_relation]
        item_j_O2 = item_j_indices[~pop_relation]
        user_embedding = self.user_embedding(user_O1)[:, 0:DICE_size]
        item_i_embedding = self.item_embedding(item_i_O1)[:, 0:DICE_size]
        item_j_embedding = self.item_embedding(item_j_O1)[:, 0:DICE_size]
        prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
        prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
        loss_interest = -1 * ((prediction_i - prediction_j).sigmoid().log().sum())
        # dict_interest = {'prediction_i': prediction_i, 
        #                  'prediction_j': prediction_j}

        user_embedding = self.user_embedding(user_O1)[:, DICE_size:]
        item_i_embedding = self.item_embedding(item_i_O1)[:, DICE_size:]
        item_j_embedding = self.item_embedding(item_j_O1)[:, DICE_size:]
        prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
        prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
        loss_popularity_1 = -1 * ((prediction_j - prediction_i).sigmoid().log().sum())
        # dict_popularity_1 = {'prediction_j': prediction_j, 
        #                      'prediction_i': prediction_i}

        user_embedding = self.user_embedding(user_O2)[:, DICE_size:]
        item_i_embedding = self.item_embedding(item_i_O2)[:, DICE_size:]
        item_j_embedding = self.item_embedding(item_j_O2)[:, DICE_size:]
        prediction_i = (user_embedding * item_i_embedding).sum(dim=-1)
        prediction_j = (user_embedding * item_j_embedding).sum(dim=-1)
        loss_popularity_2 = -1 * ((prediction_i - prediction_j).sigmoid().log().sum())
        # dict_popularity_2 = {'prediction_i': prediction_i, 
                            #  'prediction_j': prediction_j}

        # print(loss_click, loss_interest, loss_popularity_1, loss_popularity_2)
        return loss_click, loss_interest, loss_popularity_1, loss_popularity_2, loss_discrepancy, reg_loss
        # return dict_click, dict_discrepancy, dict_interest, dict_popularity_1, dict_popularity_2
    
    def predict(self, user_indices):
        user_embeddings, item_embeddings = self.get_embedding()
        user_vector = user_embeddings[user_indices]
        item_vector = item_embeddings
        
        prediction = torch.matmul(user_vector, item_vector.t())
        return prediction