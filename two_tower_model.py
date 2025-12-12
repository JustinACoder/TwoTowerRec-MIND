import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(
            self,
            user_features_dim = 385,   # the user feature dim is basically just text_embedding_dim + 1 (the history length)
            text_embedding_dim = 384,  # with our current preprocessing pipeline, that is the text embedding dimension
            #subcategory_num,
            hidden_dim=128,
            embedding_dim=64,
            subcategory_embedding_dim=16
    ):
        super(TwoTowerModel, self).__init__()

        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Item tower
        self.item_tower = nn.Sequential(
            #nn.Linear(text_embedding_dim + subcategory_embedding_dim, hidden_dim),
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Subcategory embedding
        # self.subcategory_embedding = nn.Embedding(subcategory_num, subcategory_embedding_dim)

    def forward(self, user_features, item_features, item_subcategory):
        # User tower
        user_embedding = self.user_tower(user_features)
        user_embedding = nn.functional.normalize(user_embedding, dim=-1)

        # Item tower
        # subcategory_embedded = self.subcategory_embedding(item_subcategory).squeeze(2)
        # item_embedding = self.item_tower(torch.cat([item_features, subcategory_embedded], dim=-1))
        item_embedding = self.item_tower(item_features)
        item_embedding = nn.functional.normalize(item_embedding, dim=-1)

        scores = torch.bmm(
            user_embedding.unsqueeze(1),  # (Batch, 1, Embedding_Dim)
            item_embedding.permute(0, 2, 1)  # (Batch, Embedding_Dim, Num_Items)
        ).squeeze(1)  # (Batch, Num_Items)

        return scores
