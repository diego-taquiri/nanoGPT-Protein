"""
UniRef50 dataset preprocessing script with advanced tokenization
"""

import os
import numpy as np
from tqdm import tqdm
from pyfaidx import Fasta
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, processors
from tokenizers.models import BPE

# Constants
MAX_SEQ_LENGTH = 1024  # Total sequence length including special tokens
TRUNCATION_LENGTH = 1022  # Max amino acid sequence length to allow for special tokens

# Vocabulary setup
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "]",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|", "<mask>"
]

# Create tokenizer
def create_tokenizer():
    token_to_id = {tok: ind for ind, tok in enumerate(SEQUENCE_VOCAB)}
    bpe = BPE(token_to_id, merges=[], unk_token="]")
    tokenizer = Tokenizer(bpe)
    
    # Add special tokens
    special_tokens = ["<cls>", "<pad>", "<eos>", "<mask>", "|"]
    tokenizer.add_special_tokens(special_tokens)
    
    # Configure post-processing to add special tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<cls> $A <eos>",
        special_tokens=[
            ("<cls>", token_to_id["<cls>"]),
            ("<eos>", token_to_id["<eos>"]),
        ],
    )
    
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        cls_token="<cls>",
        eos_token="<eos>",
        unk_token="]",
        mask_token="<mask>"
    )

def process_sequence(seq_str, tokenizer):
    """Tokenize and process sequence with proper padding/truncation"""
    # Tokenize with truncation and padding
    encoded = tokenizer(
        seq_str,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        return_tensors='np'
    )
    
    return encoded['input_ids'][0]  # Return as 1D array

# Setup directories
local_dir = "uniref50_shards"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = create_tokenizer()
print("Tokenizer initialized.")

# Initialize FASTA reader
print("Reading FASTA file...")
fasta_file = './data/uniref50/uniref50.fasta'
fasta = Fasta(fasta_file, as_raw=True, read_ahead=False)
print("FASTA file read successfully.")

# Process sequences and write output shards
print("Starting to process sequences...")
shard_size = int(1e5)  # Reduced shard size due to longer sequences
shard_index = 0
all_tokens = np.zeros((shard_size, MAX_SEQ_LENGTH), dtype=np.uint8)
sequence_count = 0

# Process each sequence in the FASTA file
for seq_id in tqdm(fasta.keys(), desc="Processing sequences"):
    seq = fasta[seq_id]
    tokens = process_sequence(seq[:], tokenizer)
    
    # Check if we need to start a new shard
    if sequence_count >= shard_size:
        # Write current shard
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"uniref50_{split}_{shard_index:06d}.npy")
        print(f"Writing shard {shard_index} to {filename}...")
        np.save(filename, all_tokens)
        
        # Reset for next shard
        shard_index += 1
        sequence_count = 0
        all_tokens = np.zeros((shard_size, MAX_SEQ_LENGTH), dtype=np.uint8)
    
    # Add tokens to current shard
    all_tokens[sequence_count] = tokens
    sequence_count += 1

# Write final shard if there are remaining sequences
if sequence_count > 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"uniref50_{split}_{shard_index:06d}.npy")
    print(f"Writing final shard {shard_index} to {filename}...")
    np.save(filename, all_tokens[:sequence_count])

print(f"Completed preprocessing. Total shards created: {shard_index + 1}")
