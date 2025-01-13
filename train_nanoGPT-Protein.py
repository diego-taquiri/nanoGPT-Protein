import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import pandas as pd
from pyfaidx import Fasta
import datetime

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 4096  # Increase from 256 to handle longer sequences
    vocab_size: int = 25
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        #optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, fused=use_fused)
        return optimizer


class DataLoaderLite:

    def __init__(self, 
                 B, T, 
                 process_rank, num_processes, 
                 bed_file='data/uniprot/bed_uniprot.bed', 
                 fasta_file='data/uniprot/uniprot_sprot.fasta', 
                 split='train'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = (self.process_rank == 0)

        # Prepare the Amino Acid -> integer mapping
        self.chars = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
            'B', 'Z', 'X', 'U', 'O'
        ]  # Includes standard and ambiguous amino acids
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

        # 1) Load BED and filter by split
        df_bed = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom','start','end','type'])
        df_bed = df_bed[df_bed['type'] == split]
        all_regions = df_bed[['chrom','start','end']].values.tolist()

        # 2) Round-robin partitioning so each rank sees a disjoint subset of rows
        #    e.g. rank 0 -> 0,2,4; rank 1 -> 1,3,5
        self.my_rows = [i for i in range(len(all_regions)) if i % num_processes == process_rank]
        self.regions = [all_regions[i] for i in self.my_rows]
        self.num_regions = len(self.regions)

        # 3) Open FASTA with pyfaidx
        self.fasta = Fasta(fasta_file, as_raw=True)  # as_raw=True => returns a plain Python string

        # 4) Offsets: track how far we have consumed each region (local to this rank’s subset)
        self.region_offsets = []
        for (chrom, start, end) in self.regions:
            self.region_offsets.append(start)

        # 5) current_region_idx is local to our subset
        self.current_region_idx = 0
        self.cycle_through = True  # whether to loop back after the last region

    def reset(self):
        """ Reset iteration for this DataLoader's subset of regions. """
        self.current_region_idx = 0
        for i, (chrom, start, end) in enumerate(self.regions):
            self.region_offsets[i] = start

    def _get_region_sequence(self, chrom, start, end):
        """
        Use pyfaidx to fetch (start, end) from chrom, then convert to integer tokens via self.stoi.
        With as_raw=True, self.fasta[chrom][start:end] is already a string.
        """
        seq_str = self.fasta[chrom][start:end]
        return [self.stoi[c] for c in seq_str]

    def _collect_batch_tokens(self, n_tokens_needed):
        """
        Collect up to n_tokens_needed from our local subset of regions.
        If a region is partially used, continue exactly where left off.
        """
        collected = []

        #print(f"[Rank {self.process_rank}] _collect_batch_tokens: need {n_tokens_needed} tokens")

        while len(collected) < n_tokens_needed:
            if self.current_region_idx >= self.num_regions:
                # All regions are exhausted
                if self.cycle_through:
                    #print(f"[Rank {self.process_rank}] Wrapping around to region index 0. Resetting offsets.")
                    # Reset all offsets to start processing regions again
                    for i, (chrom, start, end) in enumerate(self.regions):
                        self.region_offsets[i] = start
                    self.current_region_idx = 0
                else:
                    #print(f"[Rank {self.process_rank}] No more regions to process. Breaking.")
                    break

            chrom, region_start, region_end = self.regions[self.current_region_idx]
            offset = self.region_offsets[self.current_region_idx]
            region_length = region_end - offset

            if region_length <= 0:
                #print(f"[Rank {self.process_rank}] Region {self.current_region_idx} fully consumed, next region.")
                self.current_region_idx += 1
                continue

            needed = n_tokens_needed - len(collected)
            fetch_end = min(offset + needed, region_end)

            #print(f"[Rank {self.process_rank}] Fetch from {chrom}[{offset}:{fetch_end}] "
            #    f"(needed={needed}, have={len(collected)})")

            # Fetch tokens
            region_tokens = self._get_region_sequence(chrom, offset, fetch_end)
            collected.extend(region_tokens)

            # Update offset
            new_offset = offset + len(region_tokens)
            self.region_offsets[self.current_region_idx] = new_offset

            # If region is fully consumed, move on
            if new_offset >= region_end:
                #print(f"[Rank {self.process_rank}] Region {self.current_region_idx} consumed, advance.")
                self.current_region_idx += 1

        #print(f"[Rank {self.process_rank}] Finished collecting: {len(collected)} tokens.")
        return collected


    def next_batch(self):
        """
        Returns x, y of shape (B, T).
        We need B*T + 1 tokens total.
        """
        B, T = self.B, self.T
        n_needed = B * T + 1

        buf = self._collect_batch_tokens(n_needed)
        if len(buf) < n_needed:
            # attempt one reset
            self.reset()
            buf = self._collect_batch_tokens(n_needed)
            if len(buf) < n_needed:
                # if still short, return what we have or raise error
                pass

        buf = buf[:n_needed]
        buf_t = torch.tensor(buf, dtype=torch.long)

        x = buf_t[:-1].view(B, T)
        y = buf_t[1:].view(B, T)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # simple launch:
    # python train_gpt_dna.py
    # DDP launch for e.g. 8 GPUs:
    # torchrun --standalone --nproc_per_node=8 train_gpt_dna.py
    # run the training loop
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    init_from = "resume"  # or "scratch"
    best_val_loss = float('inf')
    save_interval = 100

    #total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
    #total_batch_size = 589824  # Adjusted to be divisible by B * T * ddp_world_size, 48 * 1024 * 2
    total_batch_size = 1179648 #2**20 #2^20, ~1M, in number of tokens,  48 * 1024 * 2 * 12
    B = 48 # micro batch size
    T = 1024 # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
    #val_loader   = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="valid") #maybe clustering to build a proper val set later

    torch.set_float32_matmul_precision('high')

    # create model
    model = GPT(GPTConfig())
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    max_lr = 2.5e-4 #2.5e-4 #6e-4
    min_lr = 1e-4#max_lr * 0.4 # * 0.1
    warmup_steps = 500 #40% #10% of the total steps
    max_steps = 127058 # 5dias #1 semana #47350 #94700 #13000 #360000
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
    
    # optimize!
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=1e-4, device=device)

    # create the log directory
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    tsv_path = os.path.join(log_dir, "training_loss.tsv")
    
    # Check if file exists; if not, create it with headers
    file_exists = os.path.exists(tsv_path)
    if not file_exists:
        with open(tsv_path, 'w') as f:
            f.write("step\tloss\tnorm\tlr\ttype\trun_checkpoint\n")
    current_run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")

    # CHECKPOINT LOADING IF RESUME
    start_step = 0
    if init_from == "resume":
        checkpoint_path = os.path.join(log_dir, "latest_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            raw_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint['step'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed training from step {start_step}, best_val_loss={best_val_loss:.4f}")
        else:
            print("No checkpoint found, starting from scratch.")

    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)
        # ----------- (1) Validation every N steps -----------
        if step % 100 == 0 and 'val_loader' in locals():
            model.eval()
            val_loader.reset()

            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20  # how many val batches we average over
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    # accumulate
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()

            if ddp:
                # average across all ranks
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

            # only rank 0 prints and logs
            if master_process:
                print(f"step {step:4d} | validation loss: {val_loss_accum:.4f}")
                # Log validation loss
                with open(tsv_path, 'a') as f:
                    f.write(f"{step}\t{val_loss_accum:.6f}\t\t\tval\t{current_run_time}_checkpoint\n")
                
                # Save if val loss improved or at save interval
                improved = val_loss_accum < best_val_loss
                if improved and (step > 0 and step % save_interval == 0):
                    # Update the best validation loss
                    best_val_loss = val_loss_accum

                    checkpoint_path = os.path.join(log_dir, "latest_checkpoint.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'best_val_loss': best_val_loss,
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

        # ----------- (2) Training Step -----------
        model.train()
        optimizer.zero_grad()
        #import code; code.interact(local=locals())
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize() # wait for the GPU to finish work
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(tsv_path, 'a') as f:
                f.write(f"{step}\t{loss_accum.item()}\t{norm:.4f}\t{lr:.4e}\ttrain\t{current_run_time}_checkpoint\n")
    if ddp:
        destroy_process_group()    