import os
import json
import numpy as np
import argparse
import re
from collections import Counter
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# === Arguments ===
parser = argparse.ArgumentParser(
    description='Train an MLP on pooled clip features and plot training curves'
)
parser.add_argument('-f', '--features_root', required=True,
                    help='Directory containing *_feat.npz files (flat directory)')
parser.add_argument('-g', '--gt_dir', required=True,
                    help='Directory containing *_metadata.json files')
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-e', '--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--step_size', type=int, default=5,
                    help='LR scheduler step size (in epochs)')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR scheduler gamma')
parser.add_argument('--plot_path', default='training_curves.png',
                    help='Path to save the loss/accuracy curves')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# === Helper to normalise clip filename ===
clip_regex = re.compile(r'^(?P<vid>[\d.]+)_clip_(?P<idx>\d+)')

def normalise_clip_name(name: str) -> str:
    base = os.path.basename(name)
    if not base.endswith('.mp4'):
        base += '.mp4'
    m = clip_regex.match(base)
    if not m:
        return base
    vid_int = int(float(m.group('vid')))
    idx = m.group('idx')
    return f"{vid_int}_clip_{idx}.mp4"

# === 1. Load GT JSONs ===
clip_to_label_raw = {}
skipped_placeholder = 0
for fn in os.listdir(args.gt_dir):
    if not fn.endswith('_metadata.json'):
        continue
    with open(os.path.join(args.gt_dir, fn), 'r') as f:
        entries = json.load(f)
    for entry in entries:
        label = str(entry.get('event', '')).strip()
        clip_raw = entry.get('clip_filename')
        if label == '-' or not clip_raw:
            skipped_placeholder += 1
            continue
        clip_to_label_raw[normalise_clip_name(clip_raw)] = label
print(f"GT entries skipped with placeholder '-' : {skipped_placeholder}")

# === Filter rare labels (<20 instances) ===
label_counts = Counter(clip_to_label_raw.values())
rare_labels = {lab for lab, cnt in label_counts.items() if cnt < 20}
if rare_labels:
    print(f"Ignoring labels (<20 instances): {sorted(rare_labels)}")
clip_to_label = {c: l for c, l in clip_to_label_raw.items() if l not in rare_labels}
print(f"Remaining labelled clips: {len(clip_to_label)}\n")

# Build class mapping
event_classes = sorted(set(clip_to_label.values()))
label2idx = {lab: i for i, lab in enumerate(event_classes)}
print(f"Classes ({len(event_classes)}): {event_classes}\n")

# === 2. Load feature files ===
X, Y = [], []
for fn in os.listdir(args.features_root):
    if not fn.endswith('_feat.npz'):
        continue
    clip_name = normalise_clip_name(fn.replace('_feat.npz', '.mp4'))
    if clip_name not in clip_to_label:
        continue
    feat_arr = np.load(os.path.join(args.features_root, fn))['feature']
    X.append(feat_arr)
    Y.append(label2idx[clip_to_label[clip_name]])

assert len(X) > 0, "No matching feature-label pairs found."
X = np.vstack(X)
Y = np.array(Y)

# === 3. Train/val split & z-score ===
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=42)
mean_vec = X_train.mean(0)
std_vec  = X_train.std(0) + 1e-8
X_train = (X_train - mean_vec) / std_vec
X_val   = (X_val   - mean_vec) / std_vec
print("Features standardised (z-score)\n")

class FeatDS(Dataset):
    def __init__(self, feats, labels):
        self.X = torch.from_numpy(feats).float()
        self.Y = torch.from_numpy(labels).long()
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_loader = DataLoader(FeatDS(X_train, Y_train), batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(FeatDS(X_val,   Y_val),   batch_size=args.batch_size)

# === 4. Model ===
input_dim = X.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(args.dropout),

    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(args.dropout),

    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(args.dropout),

    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(args.dropout),

    nn.Linear(32, len(event_classes))
).to(device)
print(f"MLP architecture: {input_dim}->256->128->64->32->{len(event_classes)} | dropout p={args.dropout}\n")

class FocalCrossEntropy(nn.Module):
    """
    Cross‑entropy focal loss (Lin et al., 2017):
        L = - α_t (1‑p_t)^γ log(p_t)

    Args
    ----
    gamma       : focusing parameter γ ≥ 0 (typical 1‑4, default 2)
    alpha       : None | scalar | Tensor of shape (C,) for per‑class weights
    reduction   : 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0,
                 alpha=None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        else:
            alpha = torch.tensor(alpha, dtype=torch.float32)
            # Register as buffer → auto‑moves with .to(device)
            self.register_buffer('alpha', alpha if alpha.ndim > 0 else alpha.unsqueeze(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (N, C) un‑normalized scores
        targets : (N,)   ground‑truth class indices
        """
        log_p = F.log_softmax(logits, dim=1)              # log‑probabilities
        p     = log_p.exp()                               # probabilities

        # Select p_t for the true class of each sample
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)   # (N,)
        p_t     = p.gather(1, targets.unsqueeze(1)).squeeze(1)       # (N,)

        # Base CE loss (negative log likelihood)
        ce_loss = -log_p_t                                           # (N,)

        # Focal modulation
        focal_term = (1.0 - p_t).pow(self.gamma)                     # (N,)
        loss = focal_term * ce_loss                                  # (N,)

        # Optional α balancing
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)                  # (N,)
            loss = alpha_t * loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:          # 'none'
            return loss
        
# ---- class counts in *class‑id* order -------------------------------
counts = torch.tensor(
    [103, 103, 75, 121, 53, 73, 35, 36, 68], dtype=torch.float32
)
num_classes = len(counts)
total       = counts.sum()

# (a) inverse‑frequency, mean‑1 normalised
alpha_invfreq = total / (num_classes * counts)

criterion = FocalCrossEntropy(gamma=2.0, alpha=alpha_invfreq).to(DEVICE)  # device = torch.device('cuda')

# ---- optimiser & scheduler stay exactly the same -------------------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)

# === 5. Training ===
train_losses, train_accs, val_accs = [], [], []
for epoch in range(1, args.epochs + 1):
    model.train(); running_loss = correct = 0; total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward(); optimizer.step()
        running_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    scheduler.step()

    train_loss = running_loss / total
    train_acc  = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval(); val_correct = 0; val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)
    val_acc = val_correct / val_total
    val_accs.append(val_acc)

    if epoch % 10 == 0 or epoch == 1:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d}: loss {train_loss:.4f} | train acc {train_acc:.4f} | val acc {val_acc:.4f} | lr {current_lr:.6f}")

# === 6. Plot curves ===
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs,   label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(args.plot_path, dpi=300)
print(f"Training curves saved to {args.plot_path}\n")

# === 7. Save artefacts ===
torch.save({
    'model_state': model.state_dict(),
    'label2idx': label2idx,
    'classes'  : event_classes,
    'feat_mean': mean_vec,
    'feat_std' : std_vec
}, 'mlp_clip_classifier.pth')
print("Training complete. Model saved to mlp_clip_classifier.pth")