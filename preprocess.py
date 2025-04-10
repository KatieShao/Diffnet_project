import numpy as np

# Paths to your files
train_path = 'data/yelp/yelp.train.rating'
val_path = 'data/yelp/yelp.val.rating'
test_path = 'data/yelp/yelp.test.rating'
links_path = 'data/yelp/yelp.links'

# --- Load training ratings ---
train_data = np.loadtxt(train_path, dtype=int)
user_ids = train_data[:, 0]
item_ids = train_data[:, 1]
# All implicit feedback is positive
labels = np.ones_like(user_ids, dtype=np.float32)

# Build consumed matrix from (user_id, item_id)
consumed_indices = np.stack((user_ids, item_ids), axis=1)
consumed_values = np.ones_like(user_ids, dtype=np.float32)

# --- Load social graph ---
link_data = np.loadtxt(links_path, dtype=int)
social_indices = link_data[:, :2]  # First two columns are user-user edges
# Weight = 1.0 for all
social_values = np.ones(social_indices.shape[0], dtype=np.float32)

# --- Load validation ratings ---
val_data = np.loadtxt(val_path, dtype=int)
val_user_ids = val_data[:, 0]
val_item_ids = val_data[:, 1]
val_labels = np.ones_like(val_user_ids, dtype=np.float32)

# --- Load test ratings ---
test_data = np.loadtxt(test_path, dtype=int)
test_user_ids = test_data[:, 0]
test_item_ids = test_data[:, 1]
test_labels = np.ones_like(test_user_ids, dtype=np.float32)

# --- Summary ---
print("Loaded training examples:", len(user_ids))
print("Validation examples:", len(val_user_ids))
print("Test examples:", len(test_user_ids))
print("User-item interactions:", consumed_indices.shape)
print("Social connections:", social_indices.shape)

# Optional: Save preprocessed arrays as .npy files
np.save('data/user_ids.npy', user_ids)
np.save('data/item_ids.npy', item_ids)
np.save('data/labels.npy', labels)
np.save('data/consumed_indices.npy', consumed_indices)
np.save('data/consumed_values.npy', consumed_values)
np.save('data/social_indices.npy', social_indices)
np.save('data/social_values.npy', social_values)

np.save('data/val_user_ids.npy', val_user_ids)
np.save('data/val_item_ids.npy', val_item_ids)
np.save('data/val_labels.npy', val_labels)

np.save('data/test_user_ids.npy', test_user_ids)
np.save('data/test_item_ids.npy', test_item_ids)
np.save('data/test_labels.npy', test_labels)
