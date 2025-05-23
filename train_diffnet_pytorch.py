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


def evaluate_model(model, conf, data_loader, social_indices, social_values, consumed_indices, consumed_values):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    with torch.no_grad():
        for user_ids, item_ids, labels in data_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            labels = labels.to(device)

            predictions = model(user_ids, item_ids,
                                social_indices, social_values,
                                consumed_indices, consumed_values)
            loss = model.calculate_loss(predictions, labels)
            total_loss += loss.item() * user_ids.size(0)
    return total_loss / len(data_loader.dataset)


def train_model(model, conf, data_dict, user_ids, item_ids, labels,
                val_user_ids=None, val_item_ids=None, val_labels=None):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)

    # Dataset and DataLoader
    train_dataset = DiffNetDataset(user_ids, item_ids, labels)
    train_loader = DataLoader(
        train_dataset, batch_size=conf.training_batch_size, shuffle=True)

    val_loader = None
    if val_user_ids is not None:
        val_dataset = DiffNetDataset(val_user_ids, val_item_ids, val_labels)
        val_loader = DataLoader(
            val_dataset, batch_size=conf.training_batch_size)

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
        model.train()
        epoch_loss = 0.0
        for batch_user, batch_item, batch_label in train_loader:
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

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        print(
            f"Epoch {epoch+1}/{conf.epochs}, Train Loss: {avg_train_loss:.4f}")

        if val_loader:
            val_loss = evaluate_model(model, conf, val_loader,
                                      social_indices, social_values,
                                      consumed_indices, consumed_values)
            print(f"Epoch {epoch+1}/{conf.epochs}, Val Loss: {val_loss:.4f}")
