import requests
import re
import pandas as pd
from collections import defaultdict
import time
from datetime import datetime, timedelta
from html import unescape
import concurrent.futures
import threading

# ---------------- CONFIGURATION ---------------- #
ARTICLES = [
    # --- GROUP 1: HIGH VANDALISM 
    'Donald Trump', 'Barack Obama', 'George W. Bush', 'Israel', 
    'Palestine (region)', 'Islam', 'Christianity', 'Scientology', 
    'Manchester United F.C.', 'Liverpool F.C.', 'Real Madrid CF', 'FC Barcelona',
    'Cristiano Ronaldo', 'Lionel Messi', 'New York Yankees', 'Boston Red Sox',
    'Justin Bieber', 'One Direction', 'Twilight (novel series)', 'Star Wars', 
    'Sonic the Hedgehog', 'My Little Pony: Friendship Is Magic', 
    'Furry fandom', 
    'PlayStation 4', 'Xbox One', 'Android (operating system)', 'iPhone', 
    'Windows 8', 'Linux', 
    'Global warming', 'Evolution', 'Flat Earth', 'Moon landing conspiracy theories',
    'Vaccination', 'Homeopathy',

    # --- GROUP 2: MEDIUM VANDALISM 
    'Taylor Swift', 'Kanye West', 'Kim Kardashian', 'Eminem', 'Rihanna',
    'Jennifer Lawrence', 'Leonardo DiCaprio', 'Johnny Depp', 'Will Smith',
    'Harry Potter', 'Game of Thrones', 'The Avengers (2012 film)', 'Avatar (2009 film)',
    'The Simpsons', 'South Park', 'Pokémon', 'Minecraft', 'Grand Theft Auto V',
    'Call of Duty', 'The Elder Scrolls V: Skyrim', 'Super Mario', 'Tetris',
    'Facebook', 'Twitter', 'YouTube', 'Google', 'Apple Inc.', 'Microsoft',
    'Amazon (company)', 'Wikipedia', 'Bitcoin', 'WikiLeaks', 'Anonymous (hacker group)',
    'Steve Jobs', 'Bill Gates', 'Mark Zuckerberg', 'Elon Musk',
    'Albert Einstein', 'Michael Jackson', 'Elvis Presley', 'Marilyn Monroe',
    'Queen Victoria', 'Abraham Lincoln', 'Martin Luther King Jr.', 'Nelson Mandela',
    'Mahatma Gandhi', 'Winston Churchill', 'Adolf Hitler', 
    'New York City', 'London', 'Paris', 'Tokyo', 'Dubai', 
    'Titanic', 'World War II', 'September 11 attacks', 'Chernobyl disaster',
    'Genetically modified organism', 

    # --- GROUP 3: LOW VANDALISM 
    'Water', 'Earth', 'Moon', 'Sun', 'Rain', 'Snow',
    'Dog', 'Cat', 'Horse', 'Lion', 'Tiger', 'Elephant', 'Bear',
    'Blue whale', 'Shark', 'Dolphin', 'Eagle', 'Penguin',
    'Tree', 'Flower', 'Rose', 'Grass', 'Forest',
    'Physics', 'Mathematics', 'Chemistry', 'Biology', 'Psychology',
    'Astronomy', 'Gravity', 'Light', 'Time', 'Atom', 'Electron',
    'Gold', 'Silver', 'Iron', 'Oxygen', 'Hydrogen', 'Carbon',
    '0', 'Pi', 'Prime number', 
    'Pizza', 'Coffee', 'Chocolate', 'Bread', 'Beer', 'Tea',
    'Guitar', 'Piano', 'Violin', 'Drum', 
    'Car', 'Bicycle', 'Train', 'Airplane', 'Ship',
    'Book', 'Computer', 'Internet', 'Color', 'Love'
]

START_DATE = "2011-01-01T00:00:00Z"
END_DATE = "2013-01-01T00:00:00Z"

headers = {"User-Agent": "GNNVandalismDetector/2.0 (Academic)", "Accept": "application/json"}

# ---------------- THREAD-SAFE GLOBAL STORAGE ---------------- #
data_lock = threading.Lock()

global_user_stats = defaultdict(lambda: { 
    "added_words": 0, "deleted_words": 0, "changed_words": 0, "reverts_done": 0,
    "added_on_users": set(), "deleted_from_users": set(),
    "edited_after_users": set(), "reverted_users": set(), "reverted_by_users": set()
})

user_edit_counts = defaultdict(int)
vandal_counts = defaultdict(int)
all_users_set = set()
all_edges_buffer = [] 

# ---------------- HELPER FUNCTIONS ---------------- #
def clean_text(text):
    if not text: return ""
    return re.sub(r"\s+", " ", unescape(text)).strip()

def get_revision_text(rev):
    return rev.get("slots", {}).get("main", {}).get("*", "") if "slots" in rev else rev.get("*", "")


# ---------------- ARTICLE PROCESSING ---------------- #
def process_single_article(article):
    
    print(f" Started: {article}")
    
    # 1. Fetch Revisions
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "prop": "revisions", "titles": article,
        "rvprop": "ids|timestamp|user|comment|content|sha1", "rvlimit": "500",  
        "rvslots": "main", "rvdir": "newer", "rvstart": START_DATE, "rvend": END_DATE, "format": "json"
    }
    
    revisions = []

    while True:
        success = False
        # RETRY LOGIC: Try 3 times before giving up
        for attempt in range(3): 
            try:
                res = requests.get(url, params=params, headers=headers, timeout=60)
                data = res.json()
                
                pages = data.get("query", {}).get("pages", {})
                for p in pages.values(): 
                    if "missing" in p:
                        print(f" Warning: '{article}' not found.")
                    revisions.extend(p.get("revisions", []))
                
                success = True
                break # Success! Exit the retry loop
            except Exception as e:
                print(f" Timeout/Error for {article} (Attempt {attempt+1}/3): {e}")
                time.sleep(5) # Wait 5 seconds before retrying

        if not success:
            print(f" FAILED to fetch {article} after 3 attempts.")
            break

        if "continue" in data: 
            params.update(data["continue"])
        else: 
            break
        
    print(f"    {article}: {len(revisions)} revs fetched. Processing")

    # 2. Local Storage 
    local_edges = []
    local_users = set()
    local_vandal_count = 0

    # 3. Analyze Interactions 
    sha_history = {} 
    prev_rev = None
    prev_user = None
    
    for i, rev in enumerate(revisions):
        user = rev.get("user")
        if not user: continue
        
        local_users.add(user)
        
        # --- Detect ClueBot Vandals ---
        if user == "ClueBot NG":
            comment = rev.get("comment", "")
            match = re.search(r"Reverting possible vandalism by\s+(?:\[\[.*?\|)?([^\s\]]+)", comment, re.IGNORECASE)
            if match:
                vandal = match.group(1).strip().replace(" ", "_").replace("User:", "")

                local_vandal_count += 1

                with data_lock:
                    vandal_counts[vandal] += 1
                    all_users_set.add(vandal)


        sha_now = rev.get("sha1")
        # (Revert Logic)
        if sha_now in sha_history:
            orig_user, orig_idx = sha_history[sha_now]
            gap = i - orig_idx
            if 0 < gap <= 15:
                with data_lock:
                    global_user_stats[user]["reverts_done"] += 1
                
                for k in range(orig_idx + 1, i):
                    victim = revisions[k].get("user")
                    if victim and victim != user:
                        local_edges.append({"src": user, "dst": victim, "type": "revert", "article": article})
                        with data_lock:
                            global_user_stats[user]["reverted_users"].add(victim)
                            global_user_stats[victim]["reverted_by_users"].add(user)
            prev_rev = rev; prev_user = user; continue

        # (Edit/Add/Delete Logic)
        if prev_rev and prev_user and prev_user != user:
            p_txt = clean_text(get_revision_text(prev_rev))
            c_txt = clean_text(get_revision_text(rev))
            p_w = set(re.findall(r"\w+", p_txt))
            c_w = set(re.findall(r"\w+", c_txt))
            added = c_w - p_w
            deleted = p_w - c_w
            
            with data_lock:
                if added: global_user_stats[user]["added_words"] += len(added)
                if deleted: global_user_stats[user]["deleted_words"] += len(deleted)
                if added and deleted: global_user_stats[user]["changed_words"] += min(len(added), len(deleted))

            if added and deleted:
                local_edges.append({"src": user, "dst": prev_user, "type": "edit_after", "article": article})
                with data_lock: global_user_stats[user]["edited_after_users"].add(prev_user)
            elif deleted:
                local_edges.append({"src": user, "dst": prev_user, "type": "delete_from", "article": article})
                with data_lock: global_user_stats[user]["deleted_from_users"].add(prev_user)
            elif added:
                local_edges.append({"src": user, "dst": prev_user, "type": "add_on", "article": article})
                with data_lock: global_user_stats[user]["added_on_users"].add(prev_user)

        sha_history[sha_now] = (user, i)
        prev_rev = rev
        prev_user = user

    # 4. GLOBAL MERGE
    with data_lock:
        for u in local_users:
            user_edit_counts[u] += 1
            all_users_set.add(u)
        all_edges_buffer.extend(local_edges)
    
    print(f"    {article}: Caught {local_vandal_count} vandals.")

    # 5. MEMORY CLEANUP 
    del revisions
 
    del sha_history
    print(f" Finished: {article}")

# ---------------- BOT DETECTION  ---------------- #
def detect_bots_batch(users_list):
    bot_set = set()
    to_check = [u for u in users_list if not (re.match(r"^\d{1,3}\.", u) or ":" in u)]
    
    print(f"\n Checking {len(to_check)} users for bot status")
    url = "https://en.wikipedia.org/w/api.php"
    
    for i in range(0, len(to_check), 50):
        batch = "|".join(to_check[i:i+50])
        try:
            res = requests.get(url, params={"action":"query", "list":"users", "ususers":batch, "usprop":"groups", "format":"json"}, headers=headers).json()
            for u in res.get("query", {}).get("users", []):
                if "bot" in u.get("groups", []):
                    bot_set.add(u["name"])
        except: pass
        time.sleep(0.5)
    return bot_set

# ---------------- MAIN  ---------------- #
def main():
    start_t = time.time()
    print(f" Starting analysis on {len(ARTICLES)} articles")
    
    # 1. Parallel Fetch & Process
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_single_article, article) for article in ARTICLES]
        concurrent.futures.wait(futures)

    print(f"\n All articles processed in {time.time()-start_t:.2f}s")
    
    # 2. Final Bot Detection
    bot_map = detect_bots_batch(list(all_users_set))
    
    # 3. Build DataFrame
    print("Building CSVs")
    users_data = []
    
    for user in all_users_set:
        if user in bot_map: continue
        
        is_anon = 1 if (re.match(r"^\d{1,3}\.", user) or ":" in user) else 0
        edits = user_edit_counts[user]
        reverts = vandal_counts[user]
        
        label = 1 if reverts > 0 else 0
        stats = global_user_stats[user]
        
        if label == 1 and edits == 0: continue # Ghost filter

        users_data.append({
            "user": user,
            "is_anonymous": is_anon, 
            "num_edits": edits, 
            "added_words": stats["added_words"], 
            "deleted_words": stats["deleted_words"],
            "changed_words": stats["changed_words"],
            "reverts_done": stats["reverts_done"],
            "num_added_on_users": len(stats["added_on_users"]),
            "num_deleted_from_users": len(stats["deleted_from_users"]),
            "num_edited_after_users": len(stats["edited_after_users"]),
            "num_reverted_users": len(stats["reverted_users"]),
            "num_reverted_by_users": len(stats["reverted_by_users"]),
            "vandal_label": label
        })
    
  
    # Save final files
    users_df = pd.DataFrame(users_data)
    edges_df = pd.DataFrame(all_edges_buffer)

    # Delete duplicate edges
    if not edges_df.empty:
        edges_df.drop_duplicates(inplace=True)

    # Save 
    users_df.to_csv("data/users_without_ratios.csv", index=False)
    edges_df.to_csv("data/edges_final.csv", index=False)
    
    
    # SUMMARY
    total_vandals = sum(1 for u in users_data if u['vandal_label'] == 1)
    
    if not edges_df.empty:
        edge_counts = edges_df['type'].value_counts()
    else:
        edge_counts = {}

    print("-" * 50)
    print(f" FINAL SUMMARY")
    print(f"   Total Articles: {len(ARTICLES)}")
    print(f"   Total Users:    {len(users_df)}")
    print(f"   Total Edges:    {len(edges_df)}")
    print(f"   TOTAL VANDALS:  {total_vandals}")
    print("-" * 50)
    print(" EDGE BREAKDOWN:")
    print(f"   • Revert: {edge_counts.get('revert', 0):,}")
    print(f"   • Add:    {edge_counts.get('add_on', 0):,}")
    print(f"   • Delete: {edge_counts.get('delete_from', 0):,}")
    print(f"   • Change: {edge_counts.get('edit_after', 0):,}")
    print("-" * 50)

if __name__ == "__main__":
    main()