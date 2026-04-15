"""
FA2 MINI PROJECT: Attention Is All You Need — Transformer Implementation

Paper Reference:
  "Attention Is All You Need" (Vaswani et al., 2017)
  https://arxiv.org/abs/1706.03762

This notebook implements:
  Full NLP pipeline: tokenisation, preprocessing, training, BLEU evaluation + BERT fine-tuning for sentiment classification
  Transformer architecture from scratch: Multi-Head Self-Attention, Positional Encoding, Encoder-Decoder, attention weight visualisation

Author  : Sharvani Arun Marne
PRN     : 123B1F061
Subject : Honors Advance Deep Learning — FA2
Date    : 15th April 2026

"""

# CELL 1 — INSTALL & IMPORTS                                     

!pip install torch torchvision --quiet
!pip install transformers datasets sacrebleu sentencepiece --quiet
!pip install matplotlib seaborn numpy tqdm --quiet
!pip install torchmetrics --quiet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch     : {torch.__version__}")
print(f"Device      : {device}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
print(f"Seed        : {SEED}")


# CELL 2 — DATA: ENGLISH ↔ FRENCH PARALLEL SENTENCES        

# Curated 300-sentence EN→FR parallel corpus (demonstrates without huge data)
EN_FR_PAIRS = [
    ("the cat sat on the mat", "le chat était assis sur le tapis"),
    ("a dog is running in the park", "un chien court dans le parc"),
    ("she loves reading books", "elle aime lire des livres"),
    ("he is eating an apple", "il mange une pomme"),
    ("the weather is beautiful today", "le temps est magnifique aujourd'hui"),
    ("i am learning deep learning", "j'apprends l'apprentissage profond"),
    ("the model predicts the output", "le modèle prédit la sortie"),
    ("attention is all you need", "l'attention est tout ce dont vous avez besoin"),
    ("the neural network is training", "le réseau de neurones s'entraîne"),
    ("they are watching a movie", "ils regardent un film"),
    ("she speaks french very well", "elle parle très bien le français"),
    ("the transformer architecture is powerful", "l'architecture transformer est puissante"),
    ("machine learning is the future", "l'apprentissage automatique est l'avenir"),
    ("the student passed the examination", "l'étudiant a réussi l'examen"),
    ("we need more training data", "nous avons besoin de plus de données d'entraînement"),
    ("the loss is decreasing slowly", "la perte diminue lentement"),
    ("gradient descent optimises the weights", "la descente de gradient optimise les poids"),
    ("the encoder reads the input", "l'encodeur lit l'entrée"),
    ("the decoder generates the output", "le décodeur génère la sortie"),
    ("multi head attention captures context", "l'attention multi-têtes capture le contexte"),
    ("positional encoding adds order information", "l'encodage positionnel ajoute des informations d'ordre"),
    ("the vocabulary size is large", "la taille du vocabulaire est grande"),
    ("batches are fed to the model", "les lots sont envoyés au modèle"),
    ("backpropagation updates the parameters", "la rétropropagation met à jour les paramètres"),
    ("the embedding layer maps tokens", "la couche d'embedding mappe les tokens"),
    ("layer normalisation stabilises training", "la normalisation par couche stabilise l'entraînement"),
    ("dropout prevents overfitting", "le dropout empêche le surapprentissage"),
    ("the softmax function produces probabilities", "la fonction softmax produit des probabilités"),
    ("residual connections help gradients flow", "les connexions résiduelles aident les gradients à circuler"),
    ("the feedforward network transforms representations", "le réseau feedforward transforme les représentations"),
    ("i am going to the market", "je vais au marché"),
    ("the children are playing outside", "les enfants jouent dehors"),
    ("she is cooking dinner tonight", "elle prépare le dîner ce soir"),
    ("the train arrives at noon", "le train arrive à midi"),
    ("he forgot his keys at home", "il a oublié ses clés à la maison"),
    ("the book is on the table", "le livre est sur la table"),
    ("we are studying computer science", "nous étudions l'informatique"),
    ("the project deadline is tomorrow", "la date limite du projet est demain"),
    ("artificial intelligence is advancing rapidly", "l'intelligence artificielle progresse rapidement"),
    ("natural language processing is fascinating", "le traitement du langage naturel est fascinant"),
    ("the conference starts at nine", "la conférence commence à neuf heures"),
    ("she sent an email to her professor", "elle a envoyé un email à son professeur"),
    ("the results are very promising", "les résultats sont très prometteurs"),
    ("we need to evaluate the model", "nous devons évaluer le modèle"),
    ("the accuracy improved after tuning", "la précision s'est améliorée après le réglage"),
    ("he presented his research findings", "il a présenté ses résultats de recherche"),
    ("the dataset contains many samples", "le jeu de données contient de nombreux échantillons"),
    ("deep learning outperforms traditional methods", "l'apprentissage profond surpasse les méthodes traditionnelles"),
    ("the paper was published last year", "le papier a été publié l'année dernière"),
    ("she graduated from university", "elle a obtenu son diplôme universitaire"),
    ("the experiment yielded good results", "l'expérience a donné de bons résultats"),
    ("i need to fix this bug", "je dois corriger ce bug"),
    ("the model is overfitting the data", "le modèle surappend les données"),
    ("regularisation improves generalisation", "la régularisation améliore la généralisation"),
    ("the validation loss is stable", "la perte de validation est stable"),
    ("he submitted the assignment on time", "il a soumis le devoir à temps"),
    ("the teacher explained the concept clearly", "l'enseignant a expliqué le concept clairement"),
    ("science and technology advance together", "la science et la technologie avancent ensemble"),
    ("the city is very crowded today", "la ville est très animée aujourd'hui"),
    ("she is writing her thesis", "elle écrit sa thèse"),
    ("the algorithm converges after many epochs", "l'algorithme converge après de nombreuses époques"),
] * 5  # Repeat to create a workable training set

random.shuffle(EN_FR_PAIRS)
split = int(0.85 * len(EN_FR_PAIRS))
TRAIN_PAIRS = EN_FR_PAIRS[:split]
VAL_PAIRS   = EN_FR_PAIRS[split:]

print(f"Total pairs  : {len(EN_FR_PAIRS)}")
print(f"Train pairs  : {len(TRAIN_PAIRS)}")
print(f"Val pairs    : {len(VAL_PAIRS)}")
print("\nSample pairs:")
for en, fr in random.sample(EN_FR_PAIRS, 4):
    print(f"  EN: {en}")
    print(f"  FR: {fr}")
    print()

# CELL 3 — NLP PIPELINE: TOKENISATION & VOCABULARY              

class Vocabulary:
    """Word-level vocabulary with special tokens."""
    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, name):
        self.name = name
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_freq = Counter()

    def build(self, sentences, min_freq=1):
        for sent in sentences:
            for word in sent.lower().split():
                self.word_freq[word] += 1
        for word, freq in self.word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        print(f"  [{self.name}] Vocabulary size: {len(self.word2idx)}")
        return self

    def encode(self, sentence, max_len=None):
        tokens = [self.word2idx.get(w, self.UNK)
                  for w in sentence.lower().split()]
        tokens = [self.SOS] + tokens + [self.EOS]
        if max_len:
            tokens = tokens[:max_len]
            tokens += [self.PAD] * (max_len - len(tokens))
        return tokens

    def decode(self, indices, skip_special=True):
        words = []
        for idx in indices:
            if skip_special and idx in (self.PAD, self.SOS, self.EOS):
                continue
            words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


print("Building vocabularies...")
src_vocab = Vocabulary("EN").build([p[0] for p in EN_FR_PAIRS])
tgt_vocab = Vocabulary("FR").build([p[1] for p in EN_FR_PAIRS])

MAX_LEN = 20

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=MAX_LEN):
        self.data = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = self.src_vocab.encode(src, self.max_len)
        tgt_ids = self.tgt_vocab.encode(tgt, self.max_len)
        return (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long))


BATCH_SIZE = 32
train_ds  = TranslationDataset(TRAIN_PAIRS, src_vocab, tgt_vocab)
val_ds    = TranslationDataset(VAL_PAIRS,   src_vocab, tgt_vocab)
train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_dl    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f"\nDataLoader:")
print(f"  Train batches : {len(train_dl)}")
print(f"  Val   batches : {len(val_dl)}")
print(f"  Src vocab size: {len(src_vocab)}")
print(f"  Tgt vocab size: {len(tgt_vocab)}")

# CELL 4 — TRANSFORMER ARCHITECTURE             
# Vaswani et al. "Attention Is All You Need" (2017)         

class PositionalEncoding(nn.Module):
    """
    Injects position information using sine and cosine functions of different
    frequencies, as defined in Vaswani et al. (2017) Section 3.5:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 4.2  Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    Equation (1) from Vaswani et al. (2017).
    Returns attention output and attention weights for visualisation.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


# 4.3  Multi-Head Attention 
class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    Section 3.2.2 of Vaswani et al. (2017).
    Stores last attention weights for external visualisation.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None   # stored for visualisation

    def split_heads(self, x):
        # x: (batch, seq, d_model) → (batch, heads, seq, d_k)
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_Q(query))
        K = self.split_heads(self.W_K(key))
        V = self.split_heads(self.W_V(value))

        if mask is not None:
            # Removed: mask = mask.unsqueeze(1) as masks are already 4D
            pass # No change needed, mask is already suitable for broadcasting

        attn_out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: (batch, heads, seq, d_k) → (batch, seq, d_model)
        B, _, S, _ = attn_out.size()
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.dropout(self.W_O(attn_out))


# 4.4  Position-wise Feed-Forward Network 
class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
    Section 3.3 of Vaswani et al. (2017).
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# 4.5  Encoder Layer 
class EncoderLayer(nn.Module):
    """
    Each encoder layer has two sub-layers:
      1. Multi-Head Self-Attention
      2. Position-wise Feed-Forward Network
    Both wrapped with residual connection + LayerNorm.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Sub-layer 1: self-attention + residual
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Sub-layer 2: FFN + residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# 4.6  Decoder Layer
class DecoderLayer(nn.Module):
    """
    Each decoder layer has three sub-layers:
      1. Masked Multi-Head Self-Attention (causal)
      2. Multi-Head Cross-Attention (attends to encoder output)
      3. Position-wise Feed-Forward Network
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 1. Masked self-attention
        sa_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(sa_out))
        # 2. Cross-attention to encoder
        ca_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(ca_out))
        # 3. FFN
        ff_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


# 4.7  Full Transformer 
class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer as in Vaswani et al. (2017).
    Default config matches the base model (d_model=256 scaled for our corpus).
    """
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, num_heads=8, num_enc_layers=3,
                 num_dec_layers=3, d_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, dropout, max_len)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_dec_layers)]
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # src: (batch, seq) — mask out PAD tokens
        return (src != 0).unsqueeze(1).unsqueeze(2)   # (B,1,1,S)

    def make_tgt_mask(self, tgt):
        B, S = tgt.size()
        pad_mask  = (tgt != 0).unsqueeze(1).unsqueeze(2)            # (B,1,1,S)
        causal    = torch.tril(torch.ones(S, S, device=tgt.device)).bool()  # (S,S)
        return pad_mask & causal.unsqueeze(0).unsqueeze(0)           # (B,1,S,S)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out  = self.encode(src, src_mask)
        dec_out  = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)   # (batch, tgt_len, tgt_vocab)


# Instantiate
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=256, num_heads=8,
    num_enc_layers=3, num_dec_layers=3,
    d_ff=512, dropout=0.1
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTransformer model built.")
print(f"Total trainable parameters: {total_params:,}")
print(f"\nModel architecture summary:")
print(f"  d_model        : 256")
print(f"  Attention heads: 8")
print(f"  Encoder layers : 3")
print(f"  Decoder layers : 3")
print(f"  FFN hidden dim : 512")
print(f"  Dropout        : 0.1")
print(f"  Src vocab      : {len(src_vocab)}")
print(f"  Tgt vocab      : {len(tgt_vocab)}")


# CELL 5 — TRAINING: NOAM SCHEDULER + LABEL SMOOTHING        

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy as used in the original Transformer paper.
    Prevents the model from becoming overconfident on training tokens.
    epsilon = 0.1 as per Vaswani et al. (2017).
    """
    def __init__(self, vocab_size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.criterion   = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence  = 1.0 - smoothing
        self.smoothing   = smoothing
        self.vocab_size  = vocab_size

    def forward(self, pred, target):
        # pred: (N, vocab) logits; target: (N,) indices
        pred = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_dist = torch.zeros_like(pred)
            smooth_dist.fill_(self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.padding_idx] = 0
            mask = (target == self.padding_idx)
            smooth_dist[mask] = 0
        return self.criterion(pred, smooth_dist)


class NoamScheduler:
    """
    Learning rate schedule from Vaswani et al. (2017) Section 5.3:
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    def __init__(self, d_model, warmup_steps=400, factor=1.0):
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self.factor        = factor
        self.step_num      = 0

    def step(self):
        self.step_num += 1
        lr = self.factor * (
            self.d_model ** -0.5
            * min(self.step_num ** -0.5,
                  self.step_num * self.warmup_steps ** -1.5)
        )
        return lr


optimizer  = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
criterion  = LabelSmoothingLoss(len(tgt_vocab), padding_idx=0, smoothing=0.1)
scheduler  = NoamScheduler(d_model=256, warmup_steps=400)

NUM_EPOCHS = 60
train_losses, val_losses, lr_history = [], [], []

print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>12}")
print("-" * 50)

for epoch in range(1, NUM_EPOCHS + 1):
    # TRAIN
    model.train()
    epoch_loss = 0
    for src_batch, tgt_batch in train_dl:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        tgt_in  = tgt_batch[:, :-1]   # decoder input (SOS…last-1)
        tgt_out = tgt_batch[:, 1:]    # decoder target (1st…EOS)

        # Noam LR update
        lr = scheduler.step()
        for g in optimizer.param_groups:
            g['lr'] = lr
        lr_history.append(lr)

        optimizer.zero_grad()
        logits = model(src_batch, tgt_in)            # (B, S-1, vocab)
        loss   = criterion(
            logits.reshape(-1, len(tgt_vocab)),
            tgt_out.reshape(-1)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    # VALIDATE 
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src_batch, tgt_batch in val_dl:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            tgt_in    = tgt_batch[:, :-1]
            tgt_out   = tgt_batch[:, 1:]
            logits    = model(src_batch, tgt_in)
            val_loss += criterion(
                logits.reshape(-1, len(tgt_vocab)),
                tgt_out.reshape(-1)
            ).item()

    avg_train = epoch_loss / len(train_dl)
    avg_val   = val_loss   / max(len(val_dl), 1)
    train_losses.append(avg_train)
    val_losses.append(avg_val)

    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:>6} | {avg_train:>10.4f} | {avg_val:>10.4f} | {lr:>12.8f}")

print("\nTraining complete.")

# CELL 6 — INFERENCE & BLEU SCORE EVALUATION                   

def greedy_translate(model, src_sentence, src_vocab, tgt_vocab,
                     max_len=MAX_LEN, device=device):
    """Greedy decoding: at each step pick the highest-probability token."""
    model.eval()
    src_ids = src_vocab.encode(src_sentence, MAX_LEN)
    src     = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src)
    enc_out  = model.encode(src, src_mask)

    tgt_ids = [tgt_vocab.SOS]
    attn_weights_list = []

    with torch.no_grad():
        for _ in range(max_len):
            tgt     = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            tgt_mask = model.make_tgt_mask(tgt)
            dec_out  = model.decode(tgt, enc_out, src_mask, tgt_mask)
            logits   = model.fc_out(dec_out[:, -1, :])
            next_id  = logits.argmax(-1).item()

            # Collect cross-attention weights from last decoder layer
            last_dec_layer = model.decoder[-1]
            w = last_dec_layer.cross_attn.attn_weights
            if w is not None:
                attn_weights_list.append(w[0].mean(0)[-1].cpu().numpy())

            tgt_ids.append(next_id)
            if next_id == tgt_vocab.EOS:
                break

    translated = tgt_vocab.decode(tgt_ids)
    return translated, attn_weights_list


def compute_bleu(references, hypotheses):
    """Corpus-level BLEU-1 and BLEU-2 (simple n-gram precision with BP)."""
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    total_match1 = total_match2 = total_hyp1 = total_hyp2 = 0
    total_ref_len = total_hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_t = ref.lower().split()
        hyp_t = hyp.lower().split()
        total_ref_len += len(ref_t)
        total_hyp_len += len(hyp_t)

        ref1, hyp1 = ngrams(ref_t, 1), ngrams(hyp_t, 1)
        match1 = sum(min(hyp1[g], ref1[g]) for g in hyp1)
        total_match1 += match1
        total_hyp1   += max(len(hyp_t), 1)

        if len(ref_t) >= 2 and len(hyp_t) >= 2:
            ref2, hyp2 = ngrams(ref_t, 2), ngrams(hyp_t, 2)
            match2 = sum(min(hyp2[g], ref2[g]) for g in hyp2)
            total_match2 += match2
            total_hyp2   += max(len(hyp_t) - 1, 1)

    bp = min(1.0, math.exp(1 - total_ref_len / max(total_hyp_len, 1)))
    bleu1 = bp * (total_match1 / max(total_hyp1, 1))
    bleu2 = bp * math.sqrt(
        (total_match1 / max(total_hyp1, 1)) *
        (total_match2 / max(total_hyp2, 1))
    )
    return round(bleu1 * 100, 2), round(bleu2 * 100, 2)


# Evaluate on validation set
references, hypotheses = [], []
print("Translating validation set...")
for en, fr_ref in VAL_PAIRS[:30]:
    translation, _ = greedy_translate(model, en, src_vocab, tgt_vocab)
    references.append(fr_ref)
    hypotheses.append(translation)

bleu1, bleu2 = compute_bleu(references, hypotheses)
print(f"\nBLEU-1 Score : {bleu1:.2f}")
print(f"BLEU-2 Score : {bleu2:.2f}")

print("\n Sample Translations")
test_sentences = [
    "attention is all you need",
    "the transformer architecture is powerful",
    "machine learning is the future",
    "the student passed the examination",
    "she loves reading books",
]
print(f"\n{'English':<40} {'Translated French'}")
print("-" * 80)
for sent in test_sentences:
    trans, _ = greedy_translate(model, sent, src_vocab, tgt_vocab)
    print(f"{sent:<40} {trans}")


# CELL 7 — VISUALISATIONS: LOSS CURVES, ATTENTION, BLEU   

# 7.1  Training Loss Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, label='Train Loss', color='#2E86AB', linewidth=2)
axes[0].plot(val_losses,   label='Val Loss',   color='#E84855', linewidth=2, linestyle='--')
axes[0].set_title('Training & Validation Loss', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(lr_history, color='#4CAF50', linewidth=1.5)
axes[1].set_title('Noam Learning Rate Schedule', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Training Step')
axes[1].set_ylabel('Learning Rate')
axes[1].grid(alpha=0.3)

bleu1_scores = []
for k in range(5, len(VAL_PAIRS[:30])+1, 5):
    refs_sub = references[:k]
    hyps_sub = hypotheses[:k]
    b1, _ = compute_bleu(refs_sub, hyps_sub)
    bleu1_scores.append(b1)
axes[2].bar(range(len(bleu1_scores)), bleu1_scores, color='#9C27B0', alpha=0.8)
axes[2].set_title('BLEU-1 on Val Subsets', fontweight='bold', fontsize=12)
axes[2].set_xlabel('Subset index (×5 sentences)')
axes[2].set_ylabel('BLEU-1 Score')
axes[2].grid(alpha=0.3, axis='y')

plt.suptitle('Transformer (Vaswani et al. 2017) — Training Metrics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: training_metrics.png")

# 7.2  Positional Encoding Heatmap
fig, ax = plt.subplots(figsize=(12, 5))
d_model, max_pos = 64, 50
pe = np.zeros((max_pos, d_model))
for pos in range(max_pos):
    for i in range(0, d_model, 2):
        pe[pos, i]   = math.sin(pos / (10000 ** (2*i/d_model)))
        pe[pos, i+1] = math.cos(pos / (10000 ** (2*i/d_model)))

im = ax.imshow(pe, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xlabel('Embedding Dimension', fontsize=11)
ax.set_ylabel('Position', fontsize=11)
ax.set_title('Positional Encoding — Sinusoidal Pattern\n(first 64 dims, 50 positions)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: positional_encoding.png")


# 7.3  Attention Weight Heatmap
demo_sentence = "attention is all you need"
_, attn_list = greedy_translate(model, demo_sentence, src_vocab, tgt_vocab)

if attn_list:
    src_tokens = ['<SOS>'] + demo_sentence.split() + ['<EOS>']
    attn_matrix = np.array(attn_list)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Show attention for each generated target position over source tokens
    disp_mat = attn_matrix[:, :len(src_tokens)]
    sns.heatmap(disp_mat,
                xticklabels=src_tokens,
                yticklabels=[f"step {i+1}" for i in range(len(attn_list))],
                cmap='Blues', linewidths=0.3, linecolor='white',
                ax=ax, vmin=0, vmax=disp_mat.max())
    ax.set_title(f'Cross-Attention Weights\n(last decoder layer, averaged over 8 heads)\nInput: "{demo_sentence}"',
                 fontweight='bold', fontsize=11)
    ax.set_xlabel('Source Token', fontsize=11)
    ax.set_ylabel('Decoder Step', fontsize=11)
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: attention_heatmap.png")


# 7.4  Multi-Head Attention — individual heads
# Run a single forward pass and extract per-head attention from encoder
model.eval()
with torch.no_grad():
    sample_src = torch.tensor(
        [src_vocab.encode(demo_sentence, MAX_LEN)], dtype=torch.long
    ).to(device)
    src_mask  = model.make_src_mask(sample_src)
    enc_x     = model.pos_enc(model.src_embed(sample_src) * math.sqrt(model.d_model))
    first_enc = model.encoder[0]
    Q = first_enc.self_attn.split_heads(first_enc.self_attn.W_Q(enc_x))
    K = first_enc.self_attn.split_heads(first_enc.self_attn.W_K(enc_x))
    V = first_enc.self_attn.split_heads(first_enc.self_attn.W_V(enc_x))
    _, head_attn = scaled_dot_product_attention(Q, K, V)
    head_attn = head_attn[0].cpu().numpy()   # (heads, seq, seq)

tokens_short = demo_sentence.split()[:8]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for h_idx, ax in enumerate(axes.flat):
    mat = head_attn[h_idx, :len(tokens_short), :len(tokens_short)]
    im  = ax.imshow(mat, cmap='Greens', vmin=0)
    ax.set_xticks(range(len(tokens_short)))
    ax.set_yticks(range(len(tokens_short)))
    ax.set_xticklabels(tokens_short, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tokens_short, fontsize=8)
    ax.set_title(f'Head {h_idx+1}', fontweight='bold', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Multi-Head Self-Attention (Encoder Layer 1) — All 8 Heads\n'
             f'Input: "{demo_sentence}"',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('multihead_attention.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: multihead_attention.png")


# CELL 8 — BERT SENTIMENT ANALYSIS NLP Pipeline

print("BERT FINE-TUNING FOR SENTIMENT ANALYSIS")

from transformers import (BertTokenizer, BertForSequenceClassification,
                           get_linear_schedule_with_warmup)

# 8.1  Dataset
SENTIMENT_DATA = [
    # Positive (1)
    ("this movie was absolutely fantastic and i loved every minute", 1),
    ("brilliant performance by all the actors, highly recommended", 1),
    ("an outstanding piece of work that left me speechless", 1),
    ("the food was delicious and the service was excellent", 1),
    ("i had a wonderful experience and will definitely come back", 1),
    ("the product exceeded my expectations in every way", 1),
    ("a masterpiece of storytelling and character development", 1),
    ("incredibly well made and deeply moving film", 1),
    ("the best restaurant i have visited in years", 1),
    ("loved the attention to detail and the beautiful visuals", 1),
    ("exceptional quality and fast delivery, very satisfied", 1),
    ("the concert was breathtaking and the crowd was amazing", 1),
    ("superb writing and a compelling plot throughout", 1),
    ("five stars for the amazing customer support team", 1),
    ("the hotel room was spotless and the staff were friendly", 1),
    ("a fantastic read that i could not put down", 1),
    ("the course material was clear and well structured", 1),
    ("genuinely impressed by the quality of the final product", 1),
    ("a joy to use and very intuitive interface", 1),
    ("perfect in every way, could not ask for more", 1),
    # Negative (0)
    ("terrible experience and i would not recommend this to anyone", 0),
    ("the worst movie i have seen in a long time", 0),
    ("extremely disappointed with the quality of the product", 0),
    ("rude staff and unacceptable waiting times", 0),
    ("the food was cold and tasteless, very disappointing", 0),
    ("broken on arrival and customer service was unhelpful", 0),
    ("boring and predictable plot with weak character development", 0),
    ("complete waste of money and time", 0),
    ("the hotel was dirty and the room smelled bad", 0),
    ("very slow delivery and the item was damaged", 0),
    ("the software crashes constantly and is full of bugs", 0),
    ("poor sound quality and terrible battery life", 0),
    ("the instructor was unprepared and hard to follow", 0),
    ("misleading description and nothing like the pictures", 0),
    ("awful smell and the packaging was completely destroyed", 0),
    ("i regret buying this, it fell apart after one day", 0),
    ("the experience was frustrating from start to finish", 0),
    ("not worth the price at all, very overrated", 0),
    ("extremely loud and uncomfortable seating arrangement", 0),
    ("the app is laggy and keeps logging me out", 0),
] * 6

random.shuffle(SENTIMENT_DATA)
s_split = int(0.8 * len(SENTIMENT_DATA))
sent_train = SENTIMENT_DATA[:s_split]
sent_val   = SENTIMENT_DATA[s_split:]
print(f"Sentiment train: {len(sent_train)} | val: {len(sent_val)}")


# 8.2  BERT Tokeniser & Dataset 
print("Loading BERT tokeniser...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data      = data
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        enc = self.tokenizer(
            text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         torch.tensor(label, dtype=torch.long)
        }

sent_train_ds = SentimentDataset(sent_train, bert_tokenizer)
sent_val_ds   = SentimentDataset(sent_val,   bert_tokenizer)
sent_train_dl = DataLoader(sent_train_ds, batch_size=16, shuffle=True)
sent_val_dl   = DataLoader(sent_val_ds,   batch_size=16, shuffle=False)


#  8.3  Fine-tune BERT
print("Loading BERT for sequence classification...")
bert_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
).to(device)

bert_optimizer = optim.AdamW(bert_model.parameters(), lr=2e-5, weight_decay=0.01)
BERT_EPOCHS    = 4
total_steps    = len(sent_train_dl) * BERT_EPOCHS
bert_scheduler = get_linear_schedule_with_warmup(
    bert_optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

bert_train_losses, bert_val_accs = [], []

print(f"\nFine-tuning BERT for {BERT_EPOCHS} epochs...")
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Accuracy':>13}")
print("-" * 36)

for epoch in range(1, BERT_EPOCHS + 1):
    bert_model.train()
    ep_loss = 0
    for batch in sent_train_dl:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)
        bert_optimizer.zero_grad()
        outputs = bert_model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        loss = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        bert_optimizer.step()
        bert_scheduler.step()
        ep_loss += loss.item()

    # Validation accuracy
    bert_model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in sent_val_dl:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            logits = bert_model(input_ids=input_ids,
                                attention_mask=attention_mask).logits
            preds  = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    avg_loss = ep_loss / len(sent_train_dl)
    acc      = correct / total * 100
    bert_train_losses.append(avg_loss)
    bert_val_accs.append(acc)
    print(f"{epoch:>6} | {avg_loss:>10.4f} | {acc:>12.2f}%")

print(f"\nFinal BERT Validation Accuracy: {bert_val_accs[-1]:.2f}%")


#  8.4  BERT Inference Demos 
def predict_sentiment(text, model=bert_model, tokenizer=bert_tokenizer):
    model.eval()
    enc = tokenizer(text, return_tensors='pt', max_length=64,
                    padding='max_length', truncation=True)
    with torch.no_grad():
        logits = model(
            input_ids      = enc['input_ids'].to(device),
            attention_mask = enc['attention_mask'].to(device)
        ).logits
    probs = F.softmax(logits, dim=-1)[0]
    label = "POSITIVE" if probs[1] > 0.5 else "NEGATIVE"
    return label, probs[1].item()

test_reviews = [
    "this is the best product I have ever purchased",
    "completely broken and the company refused to help",
    "the lecture was well explained and easy to follow",
    "boring presentation with no useful content",
    "highly recommend this to anyone looking for quality",
    "never buying from this brand again",
]
print("\n--- BERT Sentiment Predictions ---")
print(f"{'Text':<50} {'Label':>10}  {'Confidence':>12}")
print("-" * 76)
for text in test_reviews:
    label, conf = predict_sentiment(text)
    print(f"{text:<50} {label:>10}  {conf:>11.4f}")

# CELL 9 — BERT METRICS & COMBINED VISUALISATION             

# 9.1 Full Val Predictions for Confusion Matrix 
all_preds, all_labels = [], []
bert_model.eval()
with torch.no_grad():
    for batch in sent_val_dl:
        logits = bert_model(
            input_ids      = batch['input_ids'].to(device),
            attention_mask = batch['attention_mask'].to(device)
        ).logits
        preds  = logits.argmax(-1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
tp = ((all_preds == 1) & (all_labels == 1)).sum()
tn = ((all_preds == 0) & (all_labels == 0)).sum()
fp = ((all_preds == 1) & (all_labels == 0)).sum()
fn = ((all_preds == 0) & (all_labels == 1)).sum()
accuracy  = (tp + tn) / len(all_labels) * 100
precision = tp / max(tp + fp, 1) * 100
recall    = tp / max(tp + fn, 1) * 100
f1        = 2 * precision * recall / max(precision + recall, 1e-9)

print(f"\n BERT Classification Metrics ")
print(f"{'Accuracy':<20} {accuracy:.2f}%")
print(f"{'Precision':<20} {precision:.2f}%")
print(f"{'Recall':<20} {recall:.2f}%")
print(f"{'F1 Score':<20} {f1:.2f}%")

#  9.2 Plots 
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# (a) Transformer train/val loss
axes[0,0].plot(train_losses, color='#2E86AB', linewidth=2, label='Train')
axes[0,0].plot(val_losses,   color='#E84855', linewidth=2, linestyle='--', label='Val')
axes[0,0].set_title('Transformer: Train vs Val Loss', fontweight='bold')
axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

# (b) Noam LR schedule (first 2000 steps)
axes[0,1].plot(lr_history[:2000], color='#4CAF50', linewidth=1.5)
axes[0,1].set_title('Noam LR Schedule (first 2000 steps)', fontweight='bold')
axes[0,1].set_xlabel('Step'); axes[0,1].set_ylabel('LR')
axes[0,1].grid(alpha=0.3)

# (c) BLEU bar chart
metrics_names = ['BLEU-1', 'BLEU-2']
metrics_vals  = [bleu1, bleu2]
bars = axes[0,2].bar(metrics_names, metrics_vals,
                      color=['#9C27B0','#FF9800'], width=0.4, alpha=0.9)
axes[0,2].set_title('Transformer: BLEU Scores (Val)', fontweight='bold')
axes[0,2].set_ylabel('BLEU Score')
for bar, val in zip(bars, metrics_vals):
    axes[0,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   f"{val:.2f}", ha='center', fontweight='bold')

# (d) BERT loss
axes[1,0].plot(bert_train_losses, color='#F44336', linewidth=2, marker='o', markersize=6)
axes[1,0].set_title('BERT: Fine-tuning Loss', fontweight='bold')
axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('Loss')
axes[1,0].grid(alpha=0.3)

# (e) BERT accuracy
axes[1,1].plot(bert_val_accs, color='#2196F3', linewidth=2, marker='s', markersize=6)
axes[1,1].set_title('BERT: Validation Accuracy', fontweight='bold')
axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Accuracy (%)')
axes[1,1].set_ylim(0, 105)
axes[1,1].grid(alpha=0.3)
for i, v in enumerate(bert_val_accs):
    axes[1,1].text(i, v+1, f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')

# (f) Confusion matrix
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,2],
            xticklabels=['Predicted Neg', 'Predicted Pos'],
            yticklabels=['Actual Neg', 'Actual Pos'])
axes[1,2].set_title(f'BERT Confusion Matrix\n(Acc: {accuracy:.1f}%, F1: {f1:.1f}%)',
                     fontweight='bold')

plt.suptitle('FA2 — Transformer (NMT) + BERT (Sentiment) — Complete Results',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('combined_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: combined_results.png")

# CELL 10 — ARCHITECTURE DIAGRAM & FINAL SUMMARY     

# Architecture visualisation 
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')

def draw_box(ax, x, y, w, h, label, color, text_color='white', fontsize=9):
    ax.add_patch(plt.Rectangle((x, y), w, h, color=color, zorder=3,
                                linewidth=1.5, edgecolor='white'))
    ax.text(x+w/2, y+h/2, label, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold', zorder=4)

def arrow(ax, x1, y1, x2, y2, color='#555'):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=5)

# Encoder stack
draw_box(ax, 0.3, 0.5, 2.0, 1.0, 'Input\nEmbedding',      '#1976D2')
draw_box(ax, 0.3, 2.0, 2.0, 1.0, 'Positional\nEncoding',  '#1565C0')
draw_box(ax, 0.3, 3.5, 2.0, 1.2, 'Multi-Head\nSelf-Attn', '#0D47A1')
draw_box(ax, 0.3, 5.2, 2.0, 0.8, 'Add & Norm',            '#1976D2')
draw_box(ax, 0.3, 6.5, 2.0, 1.0, 'Feed-Forward\nNetwork', '#0D47A1')
draw_box(ax, 0.3, 8.0, 2.0, 0.8, 'Add & Norm',            '#1976D2')
ax.text(1.3, 9.2, 'ENCODER\n(×3 layers)', ha='center', fontweight='bold',
        fontsize=10, color='#0D47A1')

arrow(ax, 1.3, 1.5, 1.3, 2.0)
arrow(ax, 1.3, 3.0, 1.3, 3.5)
arrow(ax, 1.3, 4.7, 1.3, 5.2)
arrow(ax, 1.3, 6.0, 1.3, 6.5)
arrow(ax, 1.3, 7.5, 1.3, 8.0)

# Decoder stack
draw_box(ax, 5.5, 0.5, 2.0, 1.0, 'Output\nEmbedding',         '#C62828')
draw_box(ax, 5.5, 2.0, 2.0, 1.0, 'Positional\nEncoding',      '#B71C1C')
draw_box(ax, 5.5, 3.5, 2.0, 1.2, 'Masked MH\nSelf-Attn',      '#C62828')
draw_box(ax, 5.5, 5.2, 2.0, 0.8, 'Add & Norm',                '#E53935')
draw_box(ax, 5.5, 6.5, 2.0, 1.2, 'Cross-Attn\n(Enc→Dec)',     '#C62828')
draw_box(ax, 5.5, 8.0, 2.0, 0.8, 'Add & Norm',                '#E53935')
ax.text(6.5, 9.2, 'DECODER\n(×3 layers)', ha='center', fontweight='bold',
        fontsize=10, color='#C62828')

arrow(ax, 6.5, 1.5, 6.5, 2.0)
arrow(ax, 6.5, 3.0, 6.5, 3.5)
arrow(ax, 6.5, 4.7, 6.5, 5.2)
arrow(ax, 6.5, 6.0, 6.5, 6.5)
arrow(ax, 6.5, 7.5, 6.5, 8.0)

# Cross attention arrow encoder→decoder
ax.annotate('', xy=(5.5, 7.1), xytext=(2.3, 8.4),
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2.5), zorder=5)
ax.text(3.9, 8.3, 'encoder\noutput', ha='center', fontsize=8,
        color='#2E7D32', fontweight='bold')

# Output projection
draw_box(ax, 10.0, 3.5, 2.5, 1.2, 'Linear\nProjection',  '#37474F')
draw_box(ax, 10.0, 5.5, 2.5, 1.2, 'Softmax\n→ Token',    '#263238')
ax.text(11.25, 7.2, 'OUTPUT\nVOCAB', ha='center', fontweight='bold',
        fontsize=10, color='#263238')
arrow(ax, 7.5, 8.4, 10.0, 4.1)
arrow(ax, 11.25, 4.7, 11.25, 5.5)

plt.title('Transformer Architecture — Vaswani et al. (2017)\n"Attention Is All You Need"',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('transformer_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: transformer_architecture.png")


# Final Summary 
print("  COMPLETE RESULTS SUMMARY")
print(f"\n{'Module':<35} {'Metric':<30} {'Value':>10}")
print("-" * 77)
print(f"{'Transformer ':<35} {'Training Epochs':<30} {NUM_EPOCHS:>10}")
print(f"{'':<35} {'Final Train Loss':<30} {train_losses[-1]:>10.4f}")
print(f"{'':<35} {'Final Val Loss':<30} {val_losses[-1]:>10.4f}")
print(f"{'':<35} {'BLEU-1 Score':<30} {bleu1:>10.2f}")
print(f"{'':<35} {'BLEU-2 Score':<30} {bleu2:>10.2f}")
print(f"{'':<35} {'Total Parameters':<30} {total_params:>10,}")
print()
print(f"{'BERT Sentiment ':<35} {'Fine-tune Epochs':<30} {BERT_EPOCHS:>10}")
print(f"{'':<35} {'Validation Accuracy':<30} {accuracy:>9.2f}%")
print(f"{'':<35} {'Precision':<30} {precision:>9.2f}%")
print(f"{'':<35} {'Recall':<30} {recall:>9.2f}%")
print(f"{'':<35} {'F1 Score':<30} {f1:>9.2f}%")

print("\nOutput files generated:")
for f in ['training_metrics.png','positional_encoding.png',
          'attention_heatmap.png','multihead_attention.png',
          'combined_results.png','transformer_architecture.png']:
    print(f"  {f}")