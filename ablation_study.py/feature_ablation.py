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

print(" Starting  Feature Ablation")


# Group 1: Content/Magnitude (Raw Counts)
feats_content = [
    "num_edits", "added_words", "deleted_words", "changed_words", "reverts_done"
]

# Group 2: Network/Interaction
feats_network = [
    "num_added_on_users", "num_deleted_from_users", "num_edited_after_users", 
    "num_reverted_users", "num_reverted_by_users"
]

# Group 3: Ratios (Engineered)
feats_ratios = [
    "deletion_ratio", "aggression_ratio", "community_rejection", "revert_ratio"
]

# Group 4: Metadata
feat_anonymous = ["is_anonymous"]


experiments = [
    ("1_Content_Only", feats_content),
    ("2_Plus_Network", feats_content + feats_network),
    ("3_Plus_Ratios",  feats_content + feats_network + feats_ratios),
    ("4_Full_Model",   feats_content + feats_network + feats_ratios + feat_anonymous)
]


users_df_orig = pd.read_csv('data/users_final.csv') 
edges_df = pd.read_csv('data/edges_final.csv')


user2id = {u: i for i, u in enumerate(users_df_orig["user"].unique())}
edges_df["src"] = edges_df["src"].map(user2id)
edges_df["dst"] = edges_df["dst"].map(user2id)
edges_df = edges_df.dropna(subset=["src", "dst"])
edge_index = torch.tensor(edges_df[["src", "dst"]].values.T, dtype=torch.long)
edge_index = to_undirected(edge_index)


log_file = "data/feature_ablation_results.csv"
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "Feature_Count", "Test_F1", "Test_Prec", "Test_Rec", "Test_AUC", "Best_Threshold"])

# Helper for Threshold Tuning
def find_best_threshold_and_score(y_val, probs_val, y_test, probs_test):
    best_t, best_val_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.96, 0.01):
        s = f1_score(y_val, (probs_val >= t).astype(int), zero_division=0)
        if s > best_val_f1: best_val_f1, best_t = s, t
    preds_test = (probs_test >= best_t).astype(int)
    f1 = f1_score(y_test, preds_test, zero_division=0)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test, zero_division=0)
    auc = roc_auc_score(y_test, probs_test)
    return f1, prec, rec, auc, best_t

# --- 3. THE LOOP ---
for exp_name, current_features in experiments:
    print(f"\n Experiment: {exp_name}")
    print(f"    Using {len(current_features)} features")
    
    # Reload DF copy to start fresh
    users_df = users_df_orig.copy()
    
    # Handle NaNs
    for col in current_features:
        users_df[col] = pd.to_numeric(users_df[col], errors="coerce").fillna(0)

    # Apply Scaling (Log + Z) - Only to numericals in the current list
    # (Exclude 'is_anonymous' from log/z scaling)
    current_numericals = [f for f in current_features if f != "is_anonymous"]
    
    for col in current_numericals:
        users_df[col] = np.log1p(users_df[col])
        
    if len(current_numericals) > 0:
        scaler = StandardScaler()
        users_df[current_numericals] = scaler.fit_transform(users_df[current_numericals])

    # Create Data Object
    X = torch.tensor(users_df[current_features].values, dtype=torch.float)
    y = torch.tensor(users_df['vandal_label'].values, dtype=torch.long)
    data = Data(x=X, edge_index=edge_index, y=y)
    
    # Splits (Fixed Seed = Fair Comparison)
    train_idx_np, temp_idx_np = train_test_split(
        np.arange(data.num_nodes), test_size=0.20, stratify=data.y.numpy(), random_state=SEED
    )
    val_idx_np, test_idx_np = train_test_split(
        temp_idx_np, test_size=0.50, stratify=data.y.numpy()[temp_idx_np], random_state=SEED
    )
    
    # Masks
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool); data.train_mask[train_idx_np] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool); data.val_mask[val_idx_np] = True
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool); data.test_mask[test_idx_np] = True

    # Model Definition 
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

    # Train
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = GraphSAGE(data.num_features, HIDDEN_DIM, 2, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1]

    y_val = data.y[data.val_mask].numpy(); p_val = probs[data.val_mask].numpy()
    y_test = data.y[data.test_mask].numpy(); p_test = probs[data.test_mask].numpy()
    
    f1, prec, rec, auc, best_t = find_best_threshold_and_score(
        y_val, p_val, y_test, p_test
    )

    print(f"    Test F1: {f1:.4f} (Thresh: {best_t:.2f})")
    
    # Save
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([exp_name, len(current_features), round(f1,4), round(prec,4), round(rec,4), round(auc,4), round(best_t,2)])

