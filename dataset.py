import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from ast import literal_eval
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm

tqdm.pandas()

class RecommendationDataset(Dataset):
    def __init__(self, user_file, item_file, interaction_file, max_negs_and_pos=None):
        print("Loading user csv")
        self.users = pd.read_csv(user_file)
        print("Loading items csv")
        self.items = pd.read_csv(item_file)
        print("Loading interactions csv")
        self.interactions = pd.read_csv(interaction_file)
        print("Done loading data")
        self.max_negs_and_pos = max_negs_and_pos

        self.history_embeddings_cols = [c for c in self.users.columns if c.startswith("history_embedding_")]
        self.text_embeddings_cols = [c for c in self.items.columns if c.startswith("text_embedding_")]

        # Precompute things
        hist_emb_matrix = self.users[self.history_embeddings_cols].to_numpy(dtype=np.float32)
        hist_len = self.users["history_length"].to_numpy(dtype=np.float32).reshape(-1, 1)
        user_features_np = np.hstack([hist_emb_matrix, hist_len])
        self.user_features = torch.from_numpy(user_features_np)
        item_emb_matrix_np = self.items[self.text_embeddings_cols].to_numpy(dtype=np.float32)
        item_subcat_np = self.items['subcategory'].to_numpy(dtype=np.int32).reshape(-1, 1)
        self.item_emb_matrix = torch.from_numpy(item_emb_matrix_np)
        self.item_subcat = torch.from_numpy(item_subcat_np)

        # Map the news_ids and labels from string to list
        self.interactions['news_ids'] = self.interactions['news_ids'].progress_apply(literal_eval)
        self.interactions['labels'] = self.interactions['labels'].progress_apply(literal_eval)

        # Create mappings for user and item IDs to indices
        print("Creating user2idx dict")
        self.user2idx = {user_id: idx for idx, user_id in enumerate(self.users['user_id'].unique())}
        print("Creating item2idx dict")
        self.item2idx = {item_id: idx for idx, item_id in enumerate(self.items['news_id'].unique())}
        
    def get_user_features(self, user_id):
        user_idx = self.user2idx[user_id]
        return self.user_features[user_idx]

    def get_item_features(self, item_id):
        item_idx = self.item2idx[item_id]
        return self.item_emb_matrix[item_idx], self.item_subcat[item_idx]

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions.iloc[idx]
        user_id = interaction['user_id']
        item_ids = interaction['news_ids']
        labels = interaction['labels']

        if self.max_negs_and_pos is not None:
            max_negatives, max_positives = self.max_negs_and_pos

            # Separate positives and negatives
            pos_items = [item for item, label in zip(item_ids, labels) if label == 1]
            neg_items = [item for item, label in zip(item_ids, labels) if label == 0]

            # Limit the number of positives and negatives
            pos_items = np.random.choice(pos_items, min(len(pos_items), max_positives), replace=False).tolist()
            neg_items = np.random.choice(neg_items, min(len(neg_items), max_negatives), replace=False).tolist()

            # Combine back and create new labels
            item_ids = pos_items + neg_items
            labels = [1] * len(pos_items) + [0] * len(neg_items)

            # Shuffle the items and labels together (technically not necessary here, but good practice)
            combined = list(zip(item_ids, labels))
            np.random.shuffle(combined)
            item_ids, labels = zip(*combined)

        user_features = self.get_user_features(user_id)
        items_features = [self.get_item_features(item_id) for item_id in item_ids]
        item_embeddings = [item[0] for item in items_features]
        item_subcategories = [item[1] for item in items_features]

        return user_features, item_embeddings, item_subcategories, torch.tensor(labels, dtype=torch.int)


def impression_collate_fn(batch):
    user_features, item_embeddings, item_subcategories, labels = zip(*batch)

    user_features = torch.stack(user_features)

    # Pad item features so they all match the longest impression in this batch
    # Input list of (N_items, Dim), Output (Batch, Max_Items, Dim)
    item_embeddings = pad_sequence(
        [torch.stack(items) for items in item_embeddings],
        batch_first=True,
        padding_value=0.0
    )
    item_subcategories = pad_sequence(
        [torch.stack(subcats) for subcats in item_subcategories],
        batch_first=True,
        padding_value=0
    )

    # Pad labels similarly. Output (Batch, Max_Items)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    # Create a mask to know which items are real and which are padding
    # Output (Batch, Max_Items) - 1 for real, 0 for pad
    mask = pad_sequence(
        [torch.ones(len(lbls), dtype=torch.uint8) for lbls in labels],
        batch_first=True,
        padding_value=0
    ).bool()

    return user_features, item_embeddings, item_subcategories, labels_padded, mask
