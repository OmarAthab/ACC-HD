"""
b2_dqn.py  —  Baseline B2: reinforcement-learning handover (Double DQN / DDQN)
=============================================================================
Faithful re-implementation of the RL-handover approach of the MDPI Inventions
2025 paper, which uses a **Double Deep Q-Network (DDQN)** agent (an online
network plus a periodically-updated target network, with the double-Q update
that decouples action selection from action evaluation).

Mapping to that paper's formulation:
  * State  s_t  : the UE measurement window (RSRP/SINR per cell + current serving
                  cell), flattened — the paper's state is per-BS signal readings
                  plus the current serving-BS index.
  * Action a_t  : the index of the next serving cell.
  * Reward r_t  : signal improvement minus a handover penalty,
                  r = alpha1 * (RSRP[a] - RSRP[current]) - alpha3 * 1[a != current],
                  matching the paper's reward R = a1*dRSRQ + a2*dLoad + a3*N_HO
                  (the load term a2 is omitted because cell load is not modelled;
                  stated as a simplification in Section 5).
  * DDQN update : y = r + gamma * Q_target(s', argmax_a' Q_online(s', a')),
                  online net trained by MSE to y on the taken action; target net
                  refreshed periodically. Transitions (s_t, s_{t+1}) are formed
                  from consecutive windows in the (temporally ordered) data.

Two documented simplifications vs the paper, stated in the manuscript: (i) the
per-user-group "specialized agents" (their DBSCAN clustering) are replaced by a
single shared agent, since our evaluation uses one shared scenario; (ii) the
cell-load reward term is omitted (load is not simulated). The DDQN mechanism and
signal-based reward are otherwise faithful.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from models.train import get_device
from common.progress import ProgressPrinter


class QNet(nn.Module):
    def __init__(self, in_dim, num_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)                      # Q-values per action


def _signal_and_serving(X, num_cells):
    """From windows X (N, W, 3K) read last-step per-cell RSRP and current serving."""
    last = np.asarray(X)[:, -1, :]              # (N, 3K)
    rsrp = last[:, :num_cells]                  # (N, K) normalised RSRP
    serving = last[:, 2 * num_cells:3 * num_cells].argmax(axis=1)   # (N,)
    return rsrp.astype(np.float32), serving.astype(np.int64)


@torch.no_grad()
def _greedy_val_acc(net, X_val, y_val, device, batch_size=256):
    """Validation performance of the greedy policy: how often argmax_a Q(s,a)
    equals the true next serving cell. Used for early stopping / checkpointing,
    matching the protocol the other learned models get from train_model."""
    net.eval()
    Xf = np.asarray(X_val, dtype=np.float32).reshape(len(X_val), -1)
    Xt = torch.as_tensor(Xf)
    preds = []
    for i in range(0, len(Xt), batch_size):
        preds.append(net(Xt[i:i + batch_size].to(device)).argmax(dim=1).cpu().numpy())
    net.train()
    return float((np.concatenate(preds) == np.asarray(y_val)).mean())


def train_dqn(X_train, y_train, num_cells, epochs=120, batch_size=128,
              lr=1e-3, eps=0.1, gamma=0.9, target_sync=200,
              alpha1=1.0, alpha3=0.3, name="B2-DDQN",
              X_val=None, y_val=None, patience=12, min_delta=1e-4):
    """Train a Double DQN agent and return the online Q-network.

    When a validation set is given, the agent is trained for the FULL epoch
    budget with validation-based EARLY STOPPING and BEST-CHECKPOINT retention
    (the same fair protocol the other learned models receive via train_model),
    and the best-checkpoint network is returned. Without a validation set it
    falls back to the plain fixed-epoch loop (backward compatible).
    """
    device = get_device()
    X = np.asarray(X_train, dtype=np.float32)
    N = len(X)
    Xf = X.reshape(N, -1)
    in_dim = Xf.shape[1]
    rsrp, serving = _signal_and_serving(X, num_cells)

    online = QNet(in_dim, num_cells).to(device)
    target = QNet(in_dim, num_cells).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    mse = nn.MSELoss()

    Xt = torch.as_tensor(Xf)
    # next-state index (consecutive window); last sample points to itself (terminal-ish)
    nxt = np.arange(1, N + 1); nxt[-1] = N - 1

    step = 0
    best_acc = -1.0
    best_state = None
    no_improve = 0
    pp = ProgressPrinter(total=epochs, name=f"Training {name}")
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        ep_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx].to(device)
            qvals = online(xb)                                  # (B, K)

            # epsilon-greedy action
            greedy = qvals.argmax(dim=1).detach().cpu().numpy()
            rand = np.random.randint(0, num_cells, size=len(idx))
            explore = np.random.random(len(idx)) < eps
            actions = np.where(explore, rand, greedy)

            # reward: signal gain of chosen cell over current serving, minus HO penalty
            cur = serving[idx]
            r = alpha1 * (rsrp[idx, actions] - rsrp[idx, cur]) \
                - alpha3 * (actions != cur).astype(np.float32)

            # Double-DQN target: online selects a', target evaluates it
            with torch.no_grad():
                xb_next = Xt[nxt[idx]].to(device)
                a_next = online(xb_next).argmax(dim=1)           # online selects
                q_next = target(xb_next).gather(
                    1, a_next.unsqueeze(1)).squeeze(1)           # target evaluates
                y = torch.as_tensor(r, device=device) + gamma * q_next

            # regress the taken action's Q toward y
            target_q = qvals.detach().clone()
            target_q[np.arange(len(idx)), actions] = y
            loss = mse(qvals, target_q)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)

            step += 1
            if step % target_sync == 0:
                target.load_state_dict(online.state_dict())      # periodic sync

        # validation-based early stopping + best-checkpoint (fair protocol)
        if X_val is not None and y_val is not None:
            vacc = _greedy_val_acc(online, X_val, y_val, device)
            print(f"    [{name}] epoch {epoch:3d}/{epochs}  loss={ep_loss/N:.4f}  val_acc={vacc:.4f}",
                  flush=True)
            if vacc > best_acc + min_delta:
                best_acc = vacc
                best_state = {k: v.detach().cpu().clone() for k, v in online.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            pp.update(epoch)
            if no_improve >= patience:
                print(f"    [{name}] early stop at epoch {epoch} (best val_acc={best_acc:.4f})",
                      flush=True)
                break
        else:
            print(f"    [{name}] epoch {epoch:3d}/{epochs}  loss={ep_loss/N:.4f}",
                  flush=True)
            pp.update(epoch)
    pp.done()
    if best_state is not None:
        online.load_state_dict(best_state)       # return the best-checkpoint network
    return online


@torch.no_grad()
def predict(q, X, num_cells, batch_size=256):
    """Return softmax(Q) as probability rows so B2 goes through the same metrics."""
    device = get_device()
    Xf = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    Xt = torch.as_tensor(Xf)
    outs = []
    for i in range(0, len(Xt), batch_size):
        qv = q(Xt[i:i + batch_size].to(device))
        outs.append(torch.softmax(qv, dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0)
