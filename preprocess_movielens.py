import os
import gzip
import pickle
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split # Or use a time-based split
import argparse

# --- Import your dataset loader ---
# Assuming your loader file is named 'my_dataset_loader.py'
import datasets_loader

# --- Import necessary functions/constants (adapt paths if needed) ---
from utils import * # Need constants like USER, PRODUCT, WORD, CATEGORY, MENTION, DESCRIBED_AS, BELONG_TO, PURCHASE, DATASET_DIR, TMP_DIR
# NOTE: You might need to modify utils.py to remove BRAND, RPRODUCT and associated relations,
# or handle their absence gracefully in subsequent scripts. Let's assume modification for now.

# --- Define MovieLens specific constants ---
MOVIELENS = 'movielens' # Add this to utils.py constants if needed
DATASET_DIR[MOVIELENS] = './datasets/MovieLens' # Or your data path
TMP_DIR[MOVIELENS] = './tmp/MovieLens' # Or your tmp path

def generate_movielens_files(data_df, dataset_name):
    print(f"Processing {dataset_name} dataset...")
    data_dir = DATASET_DIR[dataset_name]
    tmp_dir = TMP_DIR[dataset_name]

    if not os.path.isdir(data_dir): os.makedirs(data_dir)
    if not os.path.isdir(tmp_dir): os.makedirs(tmp_dir)

    # --- 1. Create Mappings and Vocab Files ---
    print("Creating entity mappings and vocab files...")
    user_map = {uid: i for i, uid in enumerate(data_df['userID'].unique())}
    item_map = {iid: i for i, iid in enumerate(data_df['itemID'].unique())}

    # Genres (Categories)
    all_genres = set(g for genres in data_df['genres'].dropna() for g in genres.split(' ')) # Assumes space separation now
    genre_map = {genre: i for i, genre in enumerate(sorted(list(all_genres)))}
    category_vocab = sorted(list(all_genres))

    # Tags (Words) - Use tags associated with ratings for graph links
    user_item_tags = data_df.dropna(subset=['tag'])[['userID', 'itemID', 'tag']]
    all_tags = set(user_item_tags['tag'].unique())
    tag_map = {tag: i for i, tag in enumerate(sorted(list(all_tags)))}
    word_vocab = sorted(list(all_tags))

    # Write vocab files (.txt.gz)
    with gzip.open(os.path.join(data_dir, 'users.txt.gz'), 'wt', encoding='utf-8') as f:
        for uid in user_map: f.write(f"{uid}\n") # Write original IDs for reference if needed
    with gzip.open(os.path.join(data_dir, 'product.txt.gz'), 'wt', encoding='utf-8') as f:
        for iid in item_map: f.write(f"{iid}\n")
    with gzip.open(os.path.join(data_dir, 'category.txt.gz'), 'wt', encoding='utf-8') as f:
        for genre in category_vocab: f.write(f"{genre}\n")
    with gzip.open(os.path.join(data_dir, 'vocab.txt.gz'), 'wt', encoding='utf-8') as f: # vocab.txt.gz for words/tags
        for tag in word_vocab: f.write(f"{tag}\n")
    # Create empty files for omitted entities
    with gzip.open(os.path.join(data_dir, 'brand.txt.gz'), 'wt') as f: pass
    with gzip.open(os.path.join(data_dir, 'related_product.txt.gz'), 'wt') as f: pass

    print(f"Users: {len(user_map)}, Items: {len(item_map)}, Categories: {len(genre_map)}, Words(Tags): {len(tag_map)}")

    # Save mappings (optional, but helpful for debugging)
    mappings = {'user': user_map, 'item': item_map, 'category': genre_map, 'word': tag_map}
    with open(os.path.join(tmp_dir, 'mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)

    # --- 2. Create Relational Files ---
    print("Creating relational files...")
    # BELONG_TO (Movie -> Genre / product -> category)
    item_genres = data_df.drop_duplicates(subset=['itemID'])[['itemID', 'genres']].dropna()
    with gzip.open(os.path.join(data_dir, 'category_p_c.txt.gz'), 'wt', encoding='utf-8') as f:
        for iid_orig in item_map.keys(): # Iterate through all mapped items
            iid_idx = item_map[iid_orig]
            genres = item_genres[item_genres['itemID'] == iid_orig]['genres'].iloc[0] if iid_orig in item_genres['itemID'].values else ''
            genre_indices = [str(genre_map[g]) for g in genres.split(' ') if g in genre_map]
            f.write(" ".join(genre_indices) + "\n") # Write space-separated genre indices

    # Create empty files for omitted relations
    with gzip.open(os.path.join(data_dir, 'brand_p_b.txt.gz'), 'wt') as f: pass
    with gzip.open(os.path.join(data_dir, 'also_bought_p_p.txt.gz'), 'wt') as f: pass
    with gzip.open(os.path.join(data_dir, 'also_viewed_p_p.txt.gz'), 'wt') as f: pass
    with gzip.open(os.path.join(data_dir, 'bought_together_p_p.txt.gz'), 'wt') as f: pass


    # --- 3. Create Interaction Files (Train/Test Split) ---
    print("Creating train/test interaction files...")
    # Use only ratings for interactions, map tags separately
    ratings_df = data_df[['userID', 'itemID', 'rating', 'timestamp']].drop_duplicates(subset=['userID', 'itemID'], keep='first').copy()

    # Simple random split (consider time-based split for real scenarios)
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42, stratify=ratings_df['userID'])

    # Create tag lookup: (userID, itemID) -> list of tag_indices
    tag_lookup = defaultdict(list)
    for _, row in user_item_tags.iterrows():
        uid_idx = user_map.get(row['userID'])
        iid_idx = item_map.get(row['itemID'])
        tag_idx = tag_map.get(row['tag'])
        if uid_idx is not None and iid_idx is not None and tag_idx is not None:
            tag_lookup[(uid_idx, iid_idx)].append(str(tag_idx)) # Store as string

    # Write train.txt.gz
    with gzip.open(os.path.join(data_dir, 'train.txt.gz'), 'wt', encoding='utf-8') as f:
        for _, row in train_df.iterrows():
            uid_idx = user_map.get(row['userID'])
            iid_idx = item_map.get(row['itemID'])
            if uid_idx is not None and iid_idx is not None:
                tags_for_interaction = tag_lookup.get((uid_idx, iid_idx), [])
                f.write(f"{uid_idx}\t{iid_idx}\t{' '.join(tags_for_interaction)}\n")

    # Write test.txt.gz (no need for tags here, only user-item pairs)
    # Although the original code loads words for test too, it's not strictly necessary for label generation
    # Let's keep the format consistent for now
    with gzip.open(os.path.join(data_dir, 'test.txt.gz'), 'wt', encoding='utf-8') as f:
        for _, row in test_df.iterrows():
            uid_idx = user_map.get(row['userID'])
            iid_idx = item_map.get(row['itemID'])
            if uid_idx is not None and iid_idx is not None:
                 tags_for_interaction = tag_lookup.get((uid_idx, iid_idx), [])
                 f.write(f"{uid_idx}\t{iid_idx}\t{' '.join(tags_for_interaction)}\n") # Include tags for consistency if AmazonDataset expects it

    print("File generation complete.")


def generate_labels_movielens(dataset, mode='train'):
    # This function should now read the generated train/test.txt.gz
    review_file = '{}/{}.txt.gz'.format(DATASET_DIR[dataset], mode)
    user_products = {}  # {uid_idx: [pid_idx,...], ...}
    with gzip.open(review_file, 'r') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            arr = line.split('\t')
            user_idx = int(arr[0])
            product_idx = int(arr[1])
            # We only need user-product pairs for labels
            if user_idx not in user_products:
                user_products[user_idx] = []
            user_products[user_idx].append(product_idx)
    save_labels(dataset, user_products, mode=mode) # save_labels from utils.py


def main_preprocess_movielens():
    parser = argparse.ArgumentParser()
    # Keep dataset argument for consistency, but fix it to movielens here
    parser.add_argument('--dataset', type=str, default=MOVIELENS, help='Should be movielens.')
    args = parser.parse_args()

    if args.dataset != MOVIELENS:
        raise ValueError("This script is specifically for the MovieLens dataset.")

    # --- Load data using your loader ---
    print("Loading data using MovieLensDataset loader...")
    movielens_df = datasets_loader.loader('movielens',
                        ["userID", "itemID", "rating", "timestamp", 'title', 'genres', 'tag'],
                        1000,
                        42)
    print("Data loaded.")

    # --- Generate files needed by the pipeline ---
    generate_movielens_files(movielens_df, args.dataset)

    # --- Create the adapted "AmazonDataset" object ---
    # You might need to modify AmazonDataset or create a MovieLensDataset class
    # that inherits/adapts it to load the generated files and handle
    # the specific entities/relations (User, Product, Category, Word + Purchase, Mentions, DescribedAs, BelongTo)
    print('Load MovieLens dataset object from generated files...')
    # Example: Assuming you adapted AmazonDataset or created MovieLensPipelineDataset
    # This needs modification in data_utils.py
    # dataset = MovieLensPipelineDataset(DATASET_DIR[args.dataset]) # Pass the *directory* containing the .txt.gz files
    # save_dataset(args.dataset, dataset) # Save the processed dataset object

    # --- Create Knowledge Graph ---
    print('Create MovieLens knowledge graph from dataset object...')
    # dataset = load_dataset(args.dataset) # Load the dataset object saved above
    # kg = KnowledgeGraph(dataset) # KnowledgeGraph needs modification (see step 3)
    # kg.compute_degrees()
    # save_kg(args.dataset, kg) # Save the KG object
    print("!!! Skipping Dataset object and KG object creation/saving.")
    print("!!! Requires modifying AmazonDataset and KnowledgeGraph classes.")


    # --- Generate Train/Test Labels (using generated files) ---
    print('Generate MovieLens train/test labels.')
    generate_labels_movielens(args.dataset, 'train')
    generate_labels_movielens(args.dataset, 'test')

    print("MovieLens preprocessing finished.")
    print("NOTE: Modifications to AmazonDataset and KnowledgeGraph classes are required for full pipeline integration.")


if __name__ == '__main__':
    main_preprocess_movielens()