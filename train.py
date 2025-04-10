import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DiffNetDataset(Dataset):
    def __init__(self, user_ids, item_ids, labels):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.labels = labels

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


def train_model(model, conf, data_dict, user_ids, item_ids, labels):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)

    # Dataset and DataLoader
    dataset = DiffNetDataset(user_ids, item_ids, labels)
    data_loader = DataLoader(
        dataset, batch_size=conf.training_batch_size, shuffle=True)

    # Sparse matrices (move to device)
    social_indices = torch.tensor(
        data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT'], dtype=torch.long).to(device)
    social_values = torch.tensor(
        data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'], dtype=torch.float32).to(device)

    consumed_indices = torch.tensor(
        data_dict['CONSUMED_ITEMS_INDICES_INPUT'], dtype=torch.long).to(device)
    consumed_values = torch.tensor(
        data_dict['CONSUMED_ITEMS_VALUES_INPUT'], dtype=torch.float32).to(device)

    for epoch in range(conf.epochs):
        epoch_loss = 0.0
        for batch_user, batch_item, batch_label in data_loader:
            batch_user = batch_user.to(device)
            batch_item = batch_item.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            prediction = model(batch_user, batch_item,
                               social_indices, social_values,
                               consumed_indices, consumed_values)

            loss = model.calculate_loss(prediction, batch_label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_user.size(0)

        avg_loss = epoch_loss / len(data_loader.dataset)
        print(f"Epoch {epoch+1}/{conf.epochs}, Loss: {avg_loss:.4f}")
