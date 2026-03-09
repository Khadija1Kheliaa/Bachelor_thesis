import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================
#  WINNER CONFIGURATION
# ==========================================
HIDDEN_DIM = 128
LR = 0.01
DROPOUT = 0.5       
WEIGHT_DECAY = 0    
EPOCHS = 100
SEED = 42
AGGREGATOR = "max"
# ==========================================

print(" Starting Architecture Ablation ")


users_df = pd.read_csv('data/users_final.csv') 
edges_df = pd.read_csv('data/edges_final.csv')

# Edges
user2id = {u: i for i, u in enumerate(users_df["user"].unique())}
edges_df["src"] = edges_df["src"].map(user2id)
edges_df["dst"] = edges_df["dst"].map(user2id)
edges_df = edges_df.dropna(subset=["src", "dst"])
edge_index = torch.tensor(edges_df[["src", "dst"]].values.T, dtype=torch.long)
edge_index = to_undirected(edge_index)

# Features
count_features = [        
    "num_edits", 'added_words', 'deleted_words', 'changed_words',
    'reverts_done', 'num_added_on_users', 'num_deleted_from_users',
    'num_edited_after_users', 'num_reverted_users', 'num_reverted_by_users'
]
ratio_features = ["deletion_ratio", "aggression_ratio", "community_rejection", "revert_ratio"]
boolean_features = ["is_anonymous"]
all_features = boolean_features + count_features + ratio_features

# Handle NaNs
for col in all_features:
    users_df[col] = pd.to_numeric(users_df[col], errors="coerce").fillna(0)

# Apply Scaling
numerical_features = count_features + ratio_features
for col in numerical_features:
    users_df[col] = np.log1p(users_df[col])

scaler = StandardScaler()
users_df[numerical_features] = scaler.fit_transform(users_df[numerical_features])

# Tensor
X = torch.tensor(users_df[all_features].values, dtype=torch.float)
y = torch.tensor(users_df['vandal_label'].values, dtype=torch.long)
data = Data(x=X, edge_index=edge_index, y=y)

# Splits
train_idx_np, temp_idx_np = train_test_split(
    np.arange(data.num_nodes), test_size=0.20, stratify=data.y.numpy(), random_state=SEED
)
val_idx_np, test_idx_np = train_test_split(
    temp_idx_np, test_size=0.50, stratify=data.y.numpy()[temp_idx_np], random_state=SEED
)

# Masks 
train_idx = torch.tensor(train_idx_np, dtype=torch.long)
val_idx   = torch.tensor(val_idx_np, dtype=torch.long)
test_idx  = torch.tensor(test_idx_np, dtype=torch.long)

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool); data.train_mask[train_idx] = True
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool); data.val_mask[val_idx] = True
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool); data.test_mask[test_idx] = True

# --- 2. SETUP & HELPERS ---
log_file = "data/architecture_ablation_results.csv"
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model_Type", "Test_F1", "Test_Prec", "Test_Rec", "Test_AUC", "Best_Threshold"])

def find_best_threshold_and_score(y_val, probs_val, y_test, probs_test):
    best_t, best_val_f1 = 0.5, 0.0
    # Tune on Val
    for t in np.arange(0.1, 0.96, 0.01):
        s = f1_score(y_val, (probs_val >= t).astype(int), zero_division=0)
        if s > best_val_f1: best_val_f1, best_t = s, t
    # Test on Best Threshold
    preds_test = (probs_test >= best_t).astype(int)
    f1 = f1_score(y_test, preds_test, zero_division=0)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test, zero_division=0)
    auc = roc_auc_score(y_test, probs_test)
    return f1, prec, rec, auc, best_t

class GraphSAGE(torch.nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super().__init__()
        # Winner: Max Aggregator
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=AGGREGATOR)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=AGGREGATOR)
        self.dropout_rate = dropout_rate
    def forward(self, x, edge_index):    
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x 

# --- 3. EXPERIMENT 1: RANDOM FOREST  ---
print("\n Training Random Forest ")
# Prepare Arrays
X_train = data.x[data.train_mask].numpy()
y_train = data.y[data.train_mask].numpy()
X_val   = data.x[data.val_mask].numpy()
y_val   = data.y[data.val_mask].numpy()
X_test  = data.x[data.test_mask].numpy()
y_test  = data.y[data.test_mask].numpy()

rf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight=None)
rf.fit(X_train, y_train)

# Get Probabilities
probs_val_rf = rf.predict_proba(X_val)[:, 1]
probs_test_rf = rf.predict_proba(X_test)[:, 1]

# Tune Threshold
f1_rf, prec_rf, rec_rf, auc_rf, best_t_rf = find_best_threshold_and_score(
    y_val, probs_val_rf, y_test, probs_test_rf
)
print(f"    Random Forest Test F1: {f1_rf:.4f} (Thresh: {best_t_rf:.2f})")

with open(log_file, mode='a', newline='') as f:
    csv.writer(f).writerow(["Random_Forest", round(f1_rf,4), round(prec_rf,4), round(rec_rf,4), round(auc_rf,4), round(best_t_rf,2)])

# --- 4. EXPERIMENTS 2 & 3: GNN LOOP ---
# We define the edge configuration for both GNN experiments here
gnn_configs = [
    ("GNN_No_Edges", torch.tensor([[],[]], dtype=torch.long)),  # Empty Edges
    ("GNN_Full_With_Edges", data.edge_index)                    # Real Edges
]

criterion = torch.nn.CrossEntropyLoss()

for model_name, current_edge_index in gnn_configs:
    print(f"\n Training {model_name}...")
    
    # Reset Seeds & Model for Fairness
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = GraphSAGE(data.num_features, HIDDEN_DIM, 2, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Train
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        # Use the specific edge index for this experiment
        out = model(data.x, current_edge_index) 
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(data.x, current_edge_index)
        probs = F.softmax(out, dim=1)[:, 1]
        
    y_val_pt = data.y[data.val_mask].numpy(); probs_val = probs[data.val_mask].numpy()
    y_test_pt = data.y[data.test_mask].numpy(); probs_test = probs[data.test_mask].numpy()
    
    f1, prec, rec, auc, best_t = find_best_threshold_and_score(
        y_val_pt, probs_val, y_test_pt, probs_test
    )
    
    print(f"    {model_name} Test F1: {f1:.4f} (Thresh: {best_t:.2f})")
    
    with open(log_file, mode='a', newline='') as f:
        csv.writer(f).writerow([model_name, round(f1,4), round(prec,4), round(rec,4), round(auc,4), round(best_t,2)])
