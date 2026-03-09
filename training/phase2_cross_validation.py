import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import csv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score , precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# ==========================================
#  AUTOMATED PHASE 2 CONFIGURATION
# ==========================================

# 1. PARAMETER SEARCH SPACE (36 Combinations)
SCALING_METHODS = ["none", "log", "log_and_z"]
WEIGHT_TYPES = ["balanced", "manual", "none"]
LOSS_TYPES = ["cross_entropy", "focal"]
AGGREGATORS = ["mean", "max"]

# 2. LOCKED PARAMETERS FROM PHASE 1 
HIDDEN_DIM = 128    
LR = 0.01           
DROPOUT = 0.3       
WEIGHT_DECAY = 5e-4 

# 3. SETTINGS
EPOCHS = 100
FOLDS = 5
SEED = 42
log_file = "data/phase2_cv_results.csv"
# ==========================================

# --- 1. GLOBAL DATA LOADING (Run Once) ---


raw_users_df = pd.read_csv('data/users_final.csv') 
edges_df = pd.read_csv('data/edges_final.csv')

user2id = {u: i for i, u in enumerate(raw_users_df["user"].unique())}
edges_df["src"] = edges_df["src"].map(user2id)
edges_df["dst"] = edges_df["dst"].map(user2id)
edges_df = edges_df.dropna(subset=["src", "dst"])
edge_index = torch.tensor(edges_df[["src", "dst"]].values.T, dtype=torch.long)
edge_index = to_undirected(edge_index)

# Define Feature Groups
count_features = [        
    "num_edits", 'added_words', 'deleted_words', 'changed_words',
    'reverts_done', 'num_added_on_users', 'num_deleted_from_users',
    'num_edited_after_users', 'num_reverted_users', 'num_reverted_by_users'
]
ratio_features = ["deletion_ratio", "aggression_ratio", "community_rejection", "revert_ratio"]
boolean_features = ["is_anonymous"]


numerical_features = count_features + ratio_features
all_features_ordered = boolean_features + numerical_features

# Handle NaNs globally once
for col in all_features_ordered:
    raw_users_df[col] = pd.to_numeric(raw_users_df[col], errors="coerce").fillna(0)


# --- 2. HELPER CLASSES ---
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        inputs_class1 = inputs[:, 1] 
        bce_loss = F.binary_cross_entropy_with_logits(inputs_class1, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class GraphSAGE(torch.nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate, aggr_type):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr_type)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr_type)
        self.dropout_rate = dropout_rate
    def forward(self, x, edge_index):    
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x    


# Initialize CSV
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
           "Scaling", "Weights", "Loss", "Aggr", 
            "Val_F1_Mean", "Val_F1_Std", "Val_Prec_Mean", "Val_Prec_Std", "Val_Rec_Mean", "Val_Rec_Std",
            "Test_F1_Mean", "Test_F1_Std", "Test_Prec_Mean", "Test_Prec_Std", "Test_Rec_Mean", "Test_Rec_Std"
        ])

total_experiments = len(SCALING_METHODS) * len(WEIGHT_TYPES) * len(LOSS_TYPES) * len(AGGREGATORS)
counter = 0

print(f" Starting {total_experiments} Experiments ({FOLDS} folds each)")

# LOOP 1: SCALING (Outer loop because it modifies data)
for scaling in SCALING_METHODS:
    
    #  RESET DATA
    #  copy the raw data so previous loops don't corrupt it.
    users_df = raw_users_df.copy()
    
    # Apply Scaling Logic
    if scaling == "log":
        for col in numerical_features:
            users_df[col] = np.log1p(users_df[col])
            
    elif scaling == "log_and_z":
        for col in numerical_features:
            users_df[col] = np.log1p(users_df[col])
        scaler = StandardScaler()
        users_df[numerical_features] = scaler.fit_transform(users_df[numerical_features])
        
    # Prepare Tensor (Boolean + Scaled Numericals)
    X = torch.tensor(users_df[all_features_ordered].values, dtype=torch.float)
    y = torch.tensor(users_df['vandal_label'].values, dtype=torch.long)
    data = Data(x=X, edge_index=edge_index, y=y)

    # LOOP 2: AGGREGATOR
    for aggr in AGGREGATORS:
        # LOOP 3: LOSS
        for loss_type in LOSS_TYPES:
            # LOOP 4: WEIGHTS
            for weight_type in WEIGHT_TYPES:
                
                counter += 1
                print(f"[{counter}/{total_experiments}] Testing: {scaling} | {weight_type} | {loss_type} | {aggr} ...", end=" ")
                
                # --- START 5-FOLD CV ---
                skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
                # 1. Initialize Lists for ALL Metrics
                val_stats = {'f1': [], 'prec': [], 'rec': []}
                test_stats = {'f1': [], 'prec': [], 'rec': []}

                for fold, (train_idx_np, temp_idx_np) in enumerate(skf.split(np.zeros(data.num_nodes), data.y.numpy())):
                    # Inner Split (Temp -> Val/Test)
                    val_idx_np, test_idx_np = train_test_split(temp_idx_np, test_size=0.5, stratify=data.y.numpy()[temp_idx_np], random_state=SEED)
                    
                    # Create Masks
                    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool); train_mask[train_idx_np] = True
                    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool); val_mask[val_idx_np] = True
                    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool); test_mask[test_idx_np] = True
                    
                    # Determine Weights
                    if weight_type == "manual": final_weights = torch.tensor([1.0, 5.0], dtype=torch.float)
                    elif weight_type == "none": final_weights = None
                    else: # Balanced
                        y_train = data.y[train_mask].numpy()
                        w = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                        final_weights = torch.tensor(w, dtype=torch.float)
                    
                    # Determine Loss
                    if loss_type == "focal": criterion = FocalLoss()
                    else: criterion = torch.nn.CrossEntropyLoss(weight=final_weights)
                    
                    # Initialize Model
                    torch.manual_seed(SEED) # Reset for fairness
                    model = GraphSAGE(data.num_features, HIDDEN_DIM, 2, DROPOUT, aggr)
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                    
                    # Train
                    for epoch in range(EPOCHS):
                        model.train()
                        optimizer.zero_grad()
                        out = model(data.x, data.edge_index)
                        loss = criterion(out[train_mask], data.y[train_mask])
                        loss.backward()
                        optimizer.step()
                        
                    # Evaluate
                    model.eval()
                    with torch.no_grad():
                        out = model(data.x, data.edge_index)
                        probs = F.softmax(out, dim=1)[:, 1]
                        
                    # Find Best Threshold 
                    y_val = data.y[val_mask].numpy()
                    p_val = probs[val_mask].numpy()
                    best_t, best_val_f1 = 0.5, 0.0
                    for t in np.arange(0.1, 0.96, 0.05):
                        s = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
                        if s > best_val_f1: best_val_f1, best_t = s, t
                    
                    # Calculate Val Metrics at Best Threshold
                    val_preds = (p_val >= best_t).astype(int)
                    val_stats['f1'].append(best_val_f1)
                    val_stats['prec'].append(precision_score(y_val, val_preds, zero_division=0))
                    val_stats['rec'].append(recall_score(y_val, val_preds, zero_division=0))

                    # 2. Test: Apply Best Threshold
                    y_test = data.y[test_mask].numpy(); p_test = probs[test_mask].numpy()
                    test_preds = (p_test >= best_t).astype(int)
                    
                    test_stats['f1'].append(f1_score(y_test, test_preds, zero_division=0))
                    test_stats['prec'].append(precision_score(y_test, test_preds, zero_division=0))
                    test_stats['rec'].append(recall_score(y_test, test_preds, zero_division=0))
                
                # --- AGGREGATE STATS ---
                # Helper for Mean/Std
                def get_ms(stats_dict, key): return np.mean(stats_dict[key]), np.std(stats_dict[key])

                v_f1_m, v_f1_s = get_ms(val_stats, 'f1')
                v_p_m, v_p_s   = get_ms(val_stats, 'prec')
                v_r_m, v_r_s   = get_ms(val_stats, 'rec')

                t_f1_m, t_f1_s = get_ms(test_stats, 'f1')
                t_p_m, t_p_s   = get_ms(test_stats, 'prec')
                t_r_m, t_r_s   = get_ms(test_stats, 'rec')
                
                print(f" Val F1={v_f1_m:.4f}±{v_f1_s:.3f}")
                
                # Save to CSV 
                with open(log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        scaling, weight_type, loss_type, aggr, 
                        # Validation Stats
                        round(v_f1_m, 4), round(v_f1_s, 4),
                        round(v_p_m, 4), round(v_p_s, 4),
                        round(v_r_m, 4), round(v_r_s, 4),
                        # Test Stats
                        round(t_f1_m, 4), round(t_f1_s, 4),
                        round(t_p_m, 4), round(t_p_s, 4),
                        round(t_r_m, 4), round(t_r_s, 4)
                    ])

print(f"\n Phase 2 Complete!")