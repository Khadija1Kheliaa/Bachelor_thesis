import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# ==========================================
#  FINAL WINNER  CONFIGURATION
# ==========================================
SCALING = "log_and_z"
AGGREGATOR = "max"
WEIGHTS = "none"
LOSS = "cross_entropy"


HIDDEN_DIM = 128
LR = 0.01
DROPOUT = 0.5        
WEIGHT_DECAY = 0     

# Settings
EPOCHS = 100
SEED = 42

log_file = "data/final_test_results.csv"
# ==========================================

# --- 1. DATA PREPARATION  ---
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

print("   Applied Winner Scaling: Log + Z-Score")
for col in features_to_scale:
    users_df[col] = np.log1p(users_df[col])

scaler = StandardScaler()
users_df[features_to_scale] = scaler.fit_transform(users_df[features_to_scale])

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


# Define Model Class 
class GraphSAGE(torch.nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=AGGREGATOR)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=AGGREGATOR)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):    
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x    
    

# --- 3. TRAINING ---
print(f" Training Final Model")

# Reset Seed for absolute reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

model = GraphSAGE(data.num_features, HIDDEN_DIM, 2, DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss() 

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# --- 4. FINAL EVALUATION ---
print("\n Calculating Final Test Metrics")
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1]

# A. Find Optimal Threshold 
y_val = data.y[data.val_mask].numpy()
p_val = probs[data.val_mask].numpy()
best_t, best_f1 = 0.5, 0.0

for t in np.arange(0.1, 0.96, 0.01):
    s = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
    if s > best_f1: best_f1, best_t = s, t

print(f"    Optimal Threshold Found (Val): {best_t:.2f}")

# B. Apply to Test Set
y_test = data.y[data.test_mask].numpy()
p_test = probs[data.test_mask].numpy()
preds_test = (p_test >= best_t).astype(int)

# C. Calculate Metrics
final_f1 = f1_score(y_test, preds_test, zero_division=0)
final_prec = precision_score(y_test, preds_test, zero_division=0)
final_rec = recall_score(y_test, preds_test, zero_division=0)
final_acc = accuracy_score(y_test, preds_test)
final_auc = roc_auc_score(y_test, p_test)
tn, fp, fn, tp = confusion_matrix(y_test, preds_test).ravel()

# save results 
with open(log_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "F1_Score", "Precision", "Recall", "Accuracy", 
        "AUC_Score", "TP_Caught_Vandals", "FN_Missed_Vandals", "FP_False_Accusations"
    ])
    writer.writerow([
        round(final_f1, 4), 
        round(final_prec, 4), 
        round(final_rec, 4), 
        round(final_acc, 4),
        round(final_auc, 4),
        tp, fn, fp
    ])
        
                    
# --- 5. SUMMARY ---
print("\n" + "="*40)
print(" FINAL TEST RESULTS ")
print("="*40)
print(f"F1 Score:       {final_f1:.4f}")
print(f"Precision:      {final_prec:.4f}")
print(f"Recall:         {final_rec:.4f}")
print(f"AUC Score:      {final_auc:.4f}")
print("-" * 40)
print(f"True Vandals Caught (TP): {tp}")
print(f"Vandals Missed (FN):      {fn}")
print(f"False Accusations (FP):   {fp}")
print("="*40)
