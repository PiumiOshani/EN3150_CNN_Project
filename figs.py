# generate_figs.py
import argparse, os, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ------------------------- utils -------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def plot_curves(history, out_path, title):
    epochs = range(1, len(history["train_loss"])+1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{title} — Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path.replace(".png", "_loss.png"), dpi=200)

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train acc")
    plt.plot(epochs, history["val_acc"], label="Val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{title} — Accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=200)

def plot_confmat(y_true, y_pred, out_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.colorbar()
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(out_path, dpi=200)

# ------------------------- models -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, K=10, x1=32, m1=3, x2=64, m2=3, x3=128, d=0.3, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, x1, kernel_size=m1, padding=m1//2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(x1, x2, kernel_size=m2, padding=m2//2), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((28//4)*(28//4)*x2, x3), nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(x3, K)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def make_model(name, num_classes=10, device="cpu"):
    name = name.lower()
    if name == "custom":
        return SimpleCNN(K=num_classes).to(device), 28, False
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m.to(device), 224, True
    elif name == "googlenet":
        m = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m.to(device), 224, True
    else:
        raise ValueError("model must be one of: custom | resnet18 | googlenet")

# ------------------------- data -------------------------
def get_loaders(img_size, to_rgb, batch_size=128, seed=42):
    # MNIST: 60k train, 10k test -> we’ll re-split train into 70/15/15 overall
    base_tf = []
    # ToTensor
    base_tf += [transforms.ToTensor()]
    # replicate channels for TL
    if to_rgb:
        base_tf += [transforms.Lambda(lambda t: t.repeat(3, 1, 1))]
        # normalize by ImageNet stats
        base_tf += [transforms.Resize((img_size, img_size)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]
    else:
        # keep 1-ch, normalize to MNIST mean/std (roughly)
        base_tf += [transforms.Normalize((0.1307,), (0.3081,))]
    tf = transforms.Compose(base_tf)

    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=tf)

    # Build 70/15/15 split over the combined 70k (we’ll take from train+test to match your report)
    # Simple way: concat train+test indices, then split with stratification via targets
    X_targets = torch.cat([full_train.targets, test_set.targets], dim=0)
    N = len(X_targets)
    all_indices = np.arange(N)
    # class-balanced permutation:
    set_seed(seed)
    # stratified split by class buckets
    idx_by_class = [np.where(X_targets.numpy()==c)[0] for c in range(10)]
    for arr in idx_by_class: np.random.shuffle(arr)
    per_class = [len(a) for a in idx_by_class]
    def take_portion(portion):
        take = []
        for c, arr in enumerate(idx_by_class):
            k = int(round(per_class[c]*portion))
            take.append(arr[:k]); idx_by_class[c] = arr[k:]
        return np.concatenate(take)
    idx_train = take_portion(0.70)
    idx_val   = take_portion(0.15)
    idx_test  = np.concatenate(idx_by_class)  # remaining ~15%

    # helper to map global idx to datasets
    def subset_from_indices(indices):
        # first len(train) belong to full_train, rest to test_set
        split = len(full_train)
        take_train = indices[indices < split]
        take_test  = indices[indices >= split] - split
        parts = []
        if len(take_train): parts.append(Subset(full_train, take_train))
        if len(take_test):  parts.append(Subset(test_set,  take_test))
        if len(parts) == 2:
            from torch.utils.data import ConcatDataset
            return ConcatDataset(parts)
        return parts[0]

    ds_train = subset_from_indices(idx_train)
    ds_val   = subset_from_indices(idx_val)
    ds_test  = subset_from_indices(idx_test)

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  generator=g, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, generator=g, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, generator=g, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

# ------------------------- training -------------------------
def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def run_epoch(model, loader, opt, device, train=True):
    if train: model.train()
    else: model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ys, yhats = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train: opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if train:
                loss.backward(); opt.step()
            bs = y.size(0)
            total_loss += loss.item()*bs
            total_acc  += (logits.argmax(1)==y).float().sum().item()
            n += bs
            ys.append(y.detach().cpu().numpy())
            yhats.append(logits.argmax(1).detach().cpu().numpy())
    import numpy as np
    return total_loss/n, total_acc/n, np.concatenate(ys), np.concatenate(yhats)

def fit(model, train_loader, val_loader, device, epochs=12, lr=1e-3, wd=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_state, best_val = None, float("inf")
    patience, wait = 3, 0

    for ep in range(1, epochs+1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, opt, device, train=True)
        va_loss, va_acc, _, _ = run_epoch(model, val_loader,   opt, device, train=False)
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
        if va_loss < best_val:
            best_val = va_loss; best_state = {k:v.cpu() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    return model, history

# ------------------------- main -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom","resnet18","googlenet"], default="custom")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="figures")
    args = parser.parse_args()

    set_seed(args.seed); ensure_dir(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, img_size, to_rgb = make_model(args.model, device=device)
    train_loader, val_loader, test_loader = get_loaders(img_size, to_rgb, batch_size=args.batch, seed=args.seed)

    # Warm-up for TL models (3 epochs head-only), then unfreeze last block (simple & robust)
    if args.model in {"resnet18","googlenet"}:
        for p in model.parameters(): p.requires_grad = False
        # unfreeze classifier / last layer
        for p in model.fc.parameters(): p.requires_grad = True
        model, hist1 = fit(model, train_loader, val_loader, device, epochs=min(3, args.epochs), lr=1e-3, wd=1e-4)
        # unfreeze last stage (resnet layer4; googlenet last inception equivalent)
        for p in model.parameters(): p.requires_grad = True
        model, hist2 = fit(model, train_loader, val_loader, device, epochs=max(0, args.epochs-3), lr=1e-4, wd=1e-4)
        # merge histories
        for k in hist1: hist1[k].extend(hist2[k]); history = hist1
    else:
        model, history = fit(model, train_loader, val_loader, device, epochs=args.epochs, lr=1e-3, wd=1e-4)

    # final test pass
    _, _, y_true, y_pred = run_epoch(model, test_loader, None, device, train=False)

    # outputs
    base = {
        "custom":     "figures/custom_loss_acc.png",
        "resnet18":   "figures/resnet18_loss_acc.png",
        "googlenet":  "figures/googlenet_loss_acc.png",
    }[args.model]
    plot_curves(history, base, title=args.model.upper())
    conf_path = base.replace("loss_acc","confusion")
    plot_confmat(y_true, y_pred, conf_path, title=f"{args.model.upper()} — Confusion (Test)")

    # quick text report
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Saved: {base} and {conf_path}")

if __name__ == "__main__":
    main()
