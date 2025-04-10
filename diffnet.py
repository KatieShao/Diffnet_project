import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffNet(nn.Module):
    def __init__(self, conf):
        super(DiffNet, self).__init__()
        self.conf = conf

        # Embeddings
        self.user_embedding = nn.Parameter(
            torch.randn(conf.num_users, conf.dimension) * 0.01)
        self.item_embedding = nn.Parameter(
            torch.randn(conf.num_items, conf.dimension) * 0.01)

        # Review vector matrices (static, non-trainable)
        user_review = torch.tensor(
            np.load(conf.user_review_vector_matrix), dtype=torch.float32
        )
        item_review = torch.tensor(
            np.load(conf.item_review_vector_matrix), dtype=torch.float32
        )
        self.register_buffer('user_review_vector_matrix', user_review)
        self.register_buffer('item_review_vector_matrix', item_review)

        # Layers
        self.reduce_dimension_layer = nn.Linear(
            conf.input_review_dim, conf.dimension)
        self.user_fusion_layer = nn.Linear(conf.dimension * 2, conf.dimension)
        self.item_fusion_layer = nn.Linear(conf.dimension * 2, conf.dimension)

    def convert_distribution(self, x):
        mean = x.mean(dim=(0, 1), keepdim=True)
        std = x.std(dim=(0, 1), keepdim=True) + 1e-9
        return (x - mean) * 0.2 / std

    def forward(self, user_ids, item_ids,
                social_indices, social_values,
                consumed_indices, consumed_values):

        # Reduce review dimensions
        first_user_review = self.convert_distribution(
            self.user_review_vector_matrix)
        first_item_review = self.convert_distribution(
            self.item_review_vector_matrix)

        user_reduced = self.reduce_dimension_layer(first_user_review)
        item_reduced = self.reduce_dimension_layer(first_item_review)

        second_user_review = self.convert_distribution(user_reduced)
        second_item_review = self.convert_distribution(item_reduced)

        # Fuse embeddings
        final_item_embedding = self.item_embedding + second_item_review
        fusion_user_embedding = self.user_embedding + second_user_review

        # Sparse matrix construction
        social_shape = torch.Size([self.conf.num_users, self.conf.num_users])
        consumed_shape = torch.Size([self.conf.num_users, self.conf.num_items])

        social_sparse = torch.sparse_coo_tensor(
            indices=social_indices.T, values=social_values, size=social_shape
        ).to_dense()

        consumed_sparse = torch.sparse_coo_tensor(
            indices=consumed_indices.T, values=consumed_values, size=consumed_shape
        ).to_dense()

        user_from_social = torch.matmul(social_sparse, fusion_user_embedding)
        user_from_items = torch.matmul(consumed_sparse, final_item_embedding)

        user_embed = user_from_social + user_from_items + \
            torch.matmul(social_sparse, user_from_social)

        # Gather for prediction
        u_latent = user_embed[user_ids.squeeze()]
        i_latent = final_item_embedding[item_ids.squeeze()]

        # Dot product and sigmoid
        predict_vector = u_latent * i_latent
        prediction = torch.sigmoid(predict_vector.sum(dim=1, keepdim=True))
        return prediction

    def calculate_loss(self, prediction, labels):
        prediction = prediction.squeeze()
        return F.mse_loss(prediction, labels)
