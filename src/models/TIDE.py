import torch
from torch import nn
import torch.nn.functional as F


class TIDE(nn.Module):
    def __init__(self, backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config, graph=None):
        super().__init__()
        self.backbone = backbone  # 'MF' or 'LightGCN'
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.item_popularity = item_popularity  # tensor
        self.q = nn.Parameter(torch.ones(self.num_items + 1, ) * method_config['q'])
        self.b = nn.Parameter(torch.ones(self.num_items + 1, ) * method_config['b'])

        if self.backbone == 'MF':
            self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
            self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
            nn.init.normal_(self.user_embedding.weight, std=0.01)
            nn.init.normal_(self.item_embedding.weight, std=0.01)
        elif self.backbone == 'LightGCN':
            self.graph = graph
            self.user_embedding_0 = nn.Embedding(self.num_users, self.embedding_dim)  # the first layer
            self.item_embedding_0 = nn.Embedding(self.num_items, self.embedding_dim)  # the first layer
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
            self.user_embedding, self.item_embedding = torch.split(final_embs, [self.num_users, self.num_items])
            return self.user_embedding, self.item_embedding
    
    def forward(self, user_indices, item_i_indices, item_j_indices):
        user_embeddings, item_embeddings = self.get_embedding()
        user_vector = user_embeddings[user_indices]
        item_i_vector = item_embeddings[item_i_indices]
        item_j_vector = item_embeddings[item_j_indices]
        reg_loss = self.get_reg_loss(user_indices, item_i_indices, item_j_indices)

        item_i_popularity = self.item_popularity[item_i_indices]
        item_j_popularity = self.item_popularity[item_j_indices]

        # popularity_i_s = torch.from_numpy(self.pbd.popularity(item_i_np, timestamp)).cuda()  # 初始流行度
        item_i_popularity = F.softplus(self.q[item_i_indices]) + F.softplus(self.b[item_i_indices]) * item_i_popularity  # 先计算正样本的流行度

        # self.popularity_j = torch.from_numpy(self.pbd.popularity(item_j_np, timestamp)).cuda()
        item_j_popularity = F.softplus(self.q[item_j_indices]) + F.softplus(self.b[item_j_indices]) * item_j_popularity

        prediction_i = F.softplus((user_vector * item_i_vector).sum(dim=-1)) * torch.tanh(item_i_popularity)
        prediction_j = F.softplus((user_vector * item_j_vector).sum(dim=-1)) * torch.tanh(item_j_popularity)
        return prediction_i, prediction_j, reg_loss

    def predict(self, user_indices):
        user_embeddings, item_embeddings = self.get_embedding()
        user_vector = user_embeddings[user_indices]
        item_vector = item_embeddings
        prediction = F.softplus(torch.matmul(user_vector, item_vector.t())) * torch.tanh(self.item_popularity)
        return prediction