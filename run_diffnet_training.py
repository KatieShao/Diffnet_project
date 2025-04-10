import numpy as np
from diffnet import DiffNet  # your PyTorch model
from config import Config     # your config loader
from train_diffnet_pytorch import train_model

# Load config
conf = Config("conf/yelp_diffnet.ini")

# Load training data
user_ids = np.load("data/user_ids.npy")
item_ids = np.load("data/item_ids.npy")
labels = np.load("data/labels.npy")

# Load validation data
val_user_ids = np.load("data/val_user_ids.npy")
val_item_ids = np.load("data/val_item_ids.npy")
val_labels = np.load("data/val_labels.npy")

# Load sparse graph data
data_dict = {
    'SOCIAL_NEIGHBORS_INDICES_INPUT': np.load("data/social_indices.npy"),
    'SOCIAL_NEIGHBORS_VALUES_INPUT': np.load("data/social_values.npy"),
    'CONSUMED_ITEMS_INDICES_INPUT': np.load("data/consumed_indices.npy"),
    'CONSUMED_ITEMS_VALUES_INPUT': np.load("data/consumed_values.npy"),
}

# Build model
model = DiffNet(conf)

# Train
train_model(model, conf, data_dict, user_ids, item_ids, labels,
            val_user_ids, val_item_ids, val_labels)
