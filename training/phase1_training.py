import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import csv
import itertools
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ==========================================
#  GRID SEARCH CONFIGURATION
# ==========================================

# 1. PARAMETER LISTS 
HIDDEN_DIMS = [64, 128]
LEARNING_RATES = [0.01, 0.001]
DROPOUTS = [0.5, 0.3]
WEIGHT_DECAYS = [0, 5e-4] 

# 2.  DEFAULTS 
SCALING_METHOD = "log"
WEIGHT_TYPE = "balanced"
LOSS_TYPE = "cross_entropy"
AGGREGATOR = "mean"

# 3. SETTINGS
EPOCHS = 100
SEED = 42
log_file = "data/phase_1_results.csv"
# ==========================================

# --- 1. DATA PREPARATION ---
print(" Loading and Preprocessing Data ")
users_df = pd.read_csv('data/users_final.csv')
edges_df = pd.read_csv('data/edges_final.csv')

# Mapping
user2id = {u: i for i, u in enumerate(users_df["user"].unique())}
edges_df["src"] = edges_df["src"].map(user2id)
edges_df["dst"] = edges_df["dst"].map(user2id)
edges_df = edges_df.dropna(subset=["src", "dst"])
edge_index = torch.tensor(edges_df[["src", "dst"]].values.T, dtype=torch.long)
edge_index = to_undirected(edge_index)

count_features = [        
    "num_edits", 'added_words', 'deleted_words', 'changed_words',
    'reverts_done', 'num_added_on_users', 'num_deleted_from_users',
    'num_edited_after_users', 'num_reverted_users', 'num_reverted_by_users'
]
ratio_features = ["deletion_ratio", "aggression_ratio", "community_rejection", "revert_ratio"]

all_features = ["is_anonymous"] + count_features + ratio_features

# Handle NaNs
for col in all_features:
    users_df[col] = pd.to_numeric(users_df[col], errors="coerce").fillna(0)


features_to_scale = count_features + ratio_features

# Apply Fixed Scaling (Log)
print("   🔹 Applied Fixed Scaling: Log1p")
for col in features_to_scale:
    users_df[col] = np.log1p(users_df[col])

X = torch.tensor(users_df[all_features].values, dtype=torch.float)
y = torch.tensor(users_df['vandal_label'].values, dtype=torch.long)
data = Data(x=X, edge_index=edge_index, y=y)
num_nodes = data.num_nodes

# Splits
train_idx_np, temp_idx_np = train_test_split(
    np.arange(num_nodes), test_size=0.20, stratify=data.y.numpy(), random_state=SEED
)
val_idx_np, test_idx_np = train_test_split(
    temp_idx_np, test_size=0.50, stratify=data.y.numpy()[temp_idx_np], random_state=SEED
)

train_idx = torch.tensor(train_idx_np, dtype=torch.long)
val_idx   = torch.tensor(val_idx_np, dtype=torch.long)
test_idx  = torch.tensor(test_idx_np, dtype=torch.long)

data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
data.train_mask[train_idx] = True
data.val_mask[val_idx]     = True
data.test_mask[test_idx]   = True

# Apply Fixed Weights (Balanced)
y_train_np = data.y[data.train_mask].numpy()
classes = np.unique(y_train_np)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_np)
final_weights = torch.tensor(weights, dtype=torch.float)
print(f"   🔹 Applied Fixed Weights: Balanced {final_weights.tolist()}")

# Define Model Class 
class GraphSAGE(torch.nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):    
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x    

criterion = torch.nn.CrossEntropyLoss(weight=final_weights)

# --- 2. AUTOMATED NESTED LOOPS ---
print(f"\n Starting Grid Search: Dims={HIDDEN_DIMS} | LRs={LEARNING_RATES} | Drops={DROPOUTS} | WDs={WEIGHT_DECAYS}")
print(f" Logging to {log_file} ")

# Initialize CSV
file_exists = os.path.isfile(log_file)
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Dim", "LR", "Dropout", "WeightDecay",
        "Val_F1_Opt", "Val_Prec_Opt", "Val_Rec_Opt", "Best_Threshold", 
        "Test_F1_Opt", "Test_Prec_Opt", "Test_Rec_Opt"
        
    ])


total_combinations = len(HIDDEN_DIMS) * len(LEARNING_RATES) * len(DROPOUTS) * len(WEIGHT_DECAYS)
counter = 0

for dim in HIDDEN_DIMS:
    for lr in LEARNING_RATES:
        for drop in DROPOUTS:
            for wd in WEIGHT_DECAYS:
                counter += 1
                print(f"[{counter}/{total_combinations}] Testing: Dim={dim}, LR={lr}, Drop={drop}, WD={wd} ...", end=" ")

                #  Reset Seed for Fairness
                torch.manual_seed(SEED)
                np.random.seed(SEED)

                # Initialize Model & Optimizer 
                model = GraphSAGE(data.num_features, dim, 2, drop)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                # Training Loop
                for epoch in range(1, EPOCHS+1):
                    model.train()
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index)
                    loss = criterion(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    optimizer.step()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    probs = F.softmax(out, dim=1)[:, 1]

                # Threshold Tuning 
                y_val = data.y[data.val_mask].cpu().numpy()
                val_probs = probs[data.val_mask].cpu().numpy()
                best_threshold, best_f1_val = 0.5, 0.0

                for t in np.arange(0.1, 0.96, 0.05):
                    score = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
                    if score > best_f1_val:
                        best_f1_val, best_threshold = score, t

                val_preds_opt = (val_probs >= best_threshold).astype(int)
                val_prec = precision_score(y_val, val_preds_opt, zero_division=0)
                val_rec = recall_score(y_val, val_preds_opt, zero_division=0)

                # Final Test Metrics
                y_test = data.y[data.test_mask].cpu().numpy()
                test_probs = probs[data.test_mask].cpu().numpy()
                preds_opt = (test_probs >= best_threshold).astype(int)
                
                test_f1 = f1_score(y_test, preds_opt, zero_division=0)
                test_prec = precision_score(y_test, preds_opt, zero_division=0)
                test_rec = recall_score(y_test, preds_opt, zero_division=0)

                print(f" Val F1: {best_f1_val:.4f}")

                # Save Row to CSV
                with open(log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        dim, lr, drop, wd,
                        round(best_f1_val, 4), round(val_prec, 4), round(val_rec, 4), round(best_threshold, 2),
                        round(test_f1, 4), round(test_prec, 4), round(test_rec, 4)
                    ])

print("\n Grid Search Complete!")