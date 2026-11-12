"""
train.py
========
One generic training loop used by every neural model in the project, so the
training procedure is identical and fair across methods.

Rigor upgrades (vs code-3, which just ran a fixed 25 epochs and returned the
LAST epoch's weights):
  - Adam with weight decay (L2 regularisation).
  - ReduceLROnPlateau learning-rate schedule (halves LR when val stalls).
  - Gradient clipping (global-norm) for stable recurrent training.
  - Early stopping on the VALIDATION macro-F1 with patience.
  - BEST-checkpoint selection: we keep and return the weights of the epoch with
    the highest validation macro-F1 (not the last epoch). This makes "number of
    epochs" a validated quantity instead of a hand-picked constant.
  - Optional report_cb(epoch, val_f1) hook so a hyper-parameter search (Optuna)
    can prune unpromising trials early. Kept generic so train.py has NO optuna
    dependency.

Works on GPU automatically if available, else CPU.
"""

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from common.progress import ProgressPrinter


def get_device() -> torch.device:
    """Use the GPU if available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X, y, batch_size=128, shuffle=True):
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, loss_fn, X_train, y_train, *,
                max_epochs=100, batch_size=128, lr=1e-3, weight_decay=1e-4,
                grad_clip=5.0, patience=12, min_delta=1e-4,
                sched_patience=4, sched_factor=0.5,
                name="model", edl=False, X_val=None, y_val=None,
                print_every_seconds=30.0, verbose=True, report_cb=None):
    """Train `model`. `loss_fn(output, target[, epoch])` returns a scalar loss.

    Set edl=True when the model outputs Dirichlet alpha (evidential); then the
    loss function receives the epoch (for the annealed evidential term).

    If X_val is given, training uses early stopping + LR scheduling on the
    validation macro-F1 and returns the BEST-validation weights. If X_val is
    None it simply runs `max_epochs` (no early stopping).

    Returns: (trained model with best weights, history dict
              {'loss': [...], 'val_acc': [...], 'val_f1': [...]}).
    """
    device = get_device()
    model.to(device)
    loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_val = X_val is not None
    sched = None
    if use_val:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=sched_factor, patience=sched_patience)

    history = {"loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad_epochs = 0
    pp = ProgressPrinter(total=max_epochs, name=f"Training {name}",
                         print_every_seconds=print_every_seconds) if verbose else None

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb, epoch) if edl else loss_fn(out, yb)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_loss += loss.item() * len(xb)
            n += len(xb)
        epoch_loss /= max(1, n)
        history["loss"].append(epoch_loss)

        val_acc = float("nan")
        val_f1 = float("nan")
        if use_val:
            val_acc, val_f1 = _val_scores(model, X_val, y_val, edl=edl)
            sched.step(val_f1)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if verbose:
            msg = f"    [{name}] epoch {epoch:3d}/{max_epochs}  loss={epoch_loss:.4f}"
            if use_val:
                msg += f"  val_acc={val_acc*100:5.1f}%  val_f1={val_f1:.3f}"
            print(msg, flush=True)
            pp.update(epoch)

        # let a hyper-parameter search prune this trial early
        if report_cb is not None and use_val:
            report_cb(epoch, val_f1)     # may raise to stop the trial

        # best-checkpoint + early stopping (only meaningful with a val set)
        if use_val:
            if val_f1 > best_f1 + min_delta:
                best_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if verbose:
                        print(f"    [{name}] early stop at epoch {epoch} "
                              f"(best val_f1={best_f1:.3f} @ epoch {best_epoch})",
                              flush=True)
                    break

    if verbose and pp is not None:
        pp.done()
    if use_val:
        model.load_state_dict(best_state)          # restore BEST weights
    return model, history


@torch.no_grad()
def predict_probs(model, X, edl=False, mc_dropout_passes=0, batch_size=256):
    """Return class-probability array (N, num_classes) for inputs X.

    - edl=True: model outputs Dirichlet alpha -> convert to probabilities.
    - mc_dropout_passes>0: keep dropout ON and average several passes (MC-Dropout).
    """
    from models.heads import evidential_probs
    device = get_device()
    model.to(device)
    X_t = torch.as_tensor(X, dtype=torch.float32)

    def one_pass():
        outs = []
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(device)
            out = model(xb)
            p = evidential_probs(out) if edl else torch.softmax(out, dim=1)
            outs.append(p.cpu().numpy())
        return np.concatenate(outs, axis=0)

    if mc_dropout_passes and mc_dropout_passes > 0:
        model.train()                          # keep dropout active
        acc = np.zeros((len(X_t), _num_classes(model)), dtype=np.float64)
        for _ in range(mc_dropout_passes):
            acc += one_pass()
        return acc / mc_dropout_passes
    model.eval()
    return one_pass()


def _num_classes(model) -> int:
    """Best-effort read of the output size (used only by MC-Dropout averaging)."""
    for m in reversed(list(model.modules())):
        if hasattr(m, "out_features"):
            return m.out_features
    raise ValueError("Could not infer number of classes from model.")


@torch.no_grad()
def evaluate_accuracy(model, X, y, edl=False) -> float:
    probs = predict_probs(model, X, edl=edl)
    return float((probs.argmax(1) == np.asarray(y)).mean())


@torch.no_grad()
def _val_scores(model, X, y, edl=False):
    """Return (accuracy, macro_F1) on a validation set."""
    probs = predict_probs(model, X, edl=edl)
    pred = probs.argmax(1)
    y = np.asarray(y)
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    return acc, f1
