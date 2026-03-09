import pandas as pd
import numpy as np



users_df = pd.read_csv('data/users_without_ratios.csv')

print(f"    Loaded {len(users_df)} users.")

# 2. Denominators 
# Volume of work (Words changed/added/deleted)
total_activity_volume = (
    users_df["added_words"] + 
    users_df["deleted_words"] + 
    users_df["changed_words"] + 1
)

# Volume of Social Interaction (People I touched)
total_social_targets = (
    users_df["num_added_on_users"] + 
    users_df["num_deleted_from_users"] + 
    users_df["num_edited_after_users"] + 1
)

# 3. The Ratios

# A. Deletion Ratio (Destructiveness)
users_df["deletion_ratio"] = users_df["deleted_words"] / total_activity_volume

# B. Aggression Ratio (Widespread Attacks)
users_df["aggression_ratio"] = users_df["num_deleted_from_users"] / (users_df["num_edits"] + 1)

# C. Revert Imbalance (Rejection Rate)
users_df["community_rejection"] = users_df["num_reverted_by_users"] / total_social_targets

# D. Revert Ratio (Policing Behavior)
users_df["revert_ratio"] = users_df["reverts_done"] / (users_df["num_edits"] + 1)

# 4. Handle NaNs and Infinity
ratio_cols = ["deletion_ratio", "aggression_ratio", "community_rejection", "revert_ratio"]
for col in ratio_cols:
    users_df[col] = users_df[col].fillna(0.0)
    users_df[col] = users_df[col].replace([np.inf, -np.inf], 0.0)


users_df.to_csv('data/users_final.csv', index=False)

print(f" Success! ")
