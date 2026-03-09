import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
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

print(" Starting Threshold Analysis")


users_df = pd.read_csv('data/users_final.csv') 
edges_df = pd.read_csv('data/edges_final.csv')

# Edges
user2id = {u: i for i, u in enumerate(users_df["user"].unique())}
edges_df["src"] = edges_df["src"].map(user2id)
edges_df["dst"] = edges_df["dst"].map(user2id)
edges_df = edges_df.dropna(subset=["src", "dst"])
edge_index = torch.tensor(edges_df[["src", "dst"]].values.T, dtype=torch.long)
edge_index = to_undirected(edge_index)

# Features (Full Set)
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

# Scaling (Log + Z)
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

# --- 2. MODEL DEFINITION ---
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
torch.manual_seed(SEED)
np.random.seed(SEED)
model = GraphSAGE(data.num_features, HIDDEN_DIM, 2, DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()

print(" Training starts")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# --- 4. PREDICTION ---
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1]


y_test = data.y[data.test_mask].numpy()
probs_test = probs[data.test_mask].numpy()

# --- 5. CALCULATE METRICS FOR EVERY THRESHOLD ---
thresholds = np.arange(0.1, 0.96, 0.01)

fn_rates = []
fp_rates = []
f1_scores = []

print(" Calculating Error Rates across thresholds")

for t in thresholds:
    preds = (probs_test >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    # False Negative Rate (Miss Rate) = FN / (FN + TP)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # False Positive Rate (Fall-out) = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    f1 = f1_score(y_test, preds, zero_division=0)
    
    fn_rates.append(fnr * 100) # Convert to % for plotting
    fp_rates.append(fpr * 100)
    f1_scores.append(f1)

# Find Equal Error Rate 
diffs = np.abs(np.array(fn_rates) - np.array(fp_rates))
eer_idx = np.argmin(diffs)
eer_threshold = thresholds[eer_idx]
eer_value = (fn_rates[eer_idx] + fp_rates[eer_idx]) / 2

print(f" EER (Equal Error Rate) Found: ~{eer_value:.2f}% at Threshold {eer_threshold:.2f}")

# --- 6. PLOTTING ---
plt.figure(figsize=(18, 6))

# PLOT 1: Error Trade-off 
plt.subplot(1, 3, 1)
plt.plot(thresholds, fn_rates, label='False Negative Rate ', color='blue', linewidth=3)
plt.plot(thresholds, fp_rates, label='False Positive Rate ', color='red', linewidth=3)
plt.axvline(x=eer_threshold, color='green', linestyle='--', alpha=0.7, label=f'EER ~ {eer_value:.1f}%')
plt.title(f"Error Trade-off Curve\nEER at Thresh={eer_threshold:.2f}", fontsize=12)
plt.xlabel("Threshold")
plt.ylabel("Error %")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# PLOT 2: F1 Score Curve 
best_f1_idx = np.argmax(f1_scores)
plt.subplot(1, 3, 2)
plt.plot(thresholds, f1_scores, color='purple', linewidth=3)
plt.axvline(x=thresholds[best_f1_idx], color='orange', linestyle='--', label=f'Best F1 ({f1_scores[best_f1_idx]:.3f})')
plt.title(f"F1 Score Optimization Curve \nPeak at Thresh={thresholds[best_f1_idx]:.2f}", fontsize=12)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# PLOT 3: ROC Curve 
fpr_roc, tpr_roc, _ = roc_curve(y_test, probs_test)
roc_auc = auc(fpr_roc, tpr_roc)

plt.subplot(1, 3, 3)
plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)

# Save
plt.tight_layout()
plt.savefig("data/final_threshold_analysis.png")
print(" Plots saved to data/final_threshold_analysis.png")
plt.show()