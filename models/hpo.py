"""
hpo.py  —  principled hyper-parameter optimisation (Bayesian / Optuna)
======================================================================
Instead of hand-picking the network hyper-parameters, we SEARCH them on the VALIDATION
split with Bayesian optimisation (Optuna's TPE sampler) + median pruning.

If Optuna is not installed, this module degrades gracefully and the caller falls
back to the documented default hyper-parameters in config.TrainConfig.

References: Bergstra et al. 2011 (TPE); Akiba et al. 2019 (Optuna);
Li et al. 2018 (successive halving / median pruning).
"""

import json


def search_hyperparams(X_train, y_train, X_val, y_val,
                       num_features, num_classes, hpo_cfg, seed,
                       out_json_path=None):
    """Return a dict of best hyper-parameters (or None if Optuna is unavailable).

    Keys returned: hidden, dropout, lr, weight_decay, batch, focal_gamma.
    """
    try:
        import optuna
    except Exception as e:                       # optuna not installed
        print(f"  [HPO] Optuna not available ({e}); using default hyper-parameters.")
        return None

    from common.seeding import set_all_seeds
    from baselines.uq_baselines import SoftmaxGRU
    from models.heads import focal_loss
    from models.train import train_model

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", list(hpo_cfg.hidden_choices))
        dropout = trial.suggest_float("dropout", *hpo_cfg.dropout_range)
        lr = trial.suggest_float("lr", *hpo_cfg.lr_range, log=True)
        wd = trial.suggest_float("weight_decay", *hpo_cfg.weight_decay_range, log=True)
        batch = trial.suggest_categorical("batch", list(hpo_cfg.batch_choices))
        gamma = trial.suggest_float("focal_gamma", *hpo_cfg.focal_gamma_range)

        set_all_seeds(seed)                      # same init per trial -> fair search
        model = SoftmaxGRU(num_features, num_classes, hidden, dropout)

        def report_cb(epoch, val_f1):
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, hist = train_model(
            model, lambda o, t: focal_loss(o, t, gamma),
            X_train, y_train,
            max_epochs=hpo_cfg.max_epochs_per_trial,
            batch_size=batch, lr=lr, weight_decay=wd,
            patience=hpo_cfg.patience_per_trial,
            name=f"hpo-t{trial.number}", edl=False,
            X_val=X_val, y_val=y_val,
            verbose=False, report_cb=report_cb)
        return max(hist["val_f1"])               # best validation macro-F1

    sampler = optuna.samplers.TPESampler(seed=seed)          # Bayesian (TPE)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)   # early pruning
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    timeout = hpo_cfg.timeout_seconds or None
    print(f"  [HPO] Bayesian search: up to {hpo_cfg.n_trials} trials "
          f"({hpo_cfg.max_epochs_per_trial} epochs/trial, median pruning)...")
    study.optimize(objective, n_trials=hpo_cfg.n_trials, timeout=timeout)

    best = dict(study.best_params)
    print(f"  [HPO] best validation macro-F1 = {study.best_value:.4f}")
    print(f"  [HPO] best params = {best}")

    if out_json_path is not None:
        payload = {
            "best_val_macro_f1": float(study.best_value),
            "best_params": best,
            "n_trials_run": len(study.trials),
            "search_space": {
                "hidden_choices": list(hpo_cfg.hidden_choices),
                "dropout_range": list(hpo_cfg.dropout_range),
                "lr_range": list(hpo_cfg.lr_range),
                "weight_decay_range": list(hpo_cfg.weight_decay_range),
                "batch_choices": list(hpo_cfg.batch_choices),
                "focal_gamma_range": list(hpo_cfg.focal_gamma_range),
            },
            "sampler": "TPE (Bayesian)", "pruner": "MedianPruner",
        }
        try:
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"  [HPO] search record saved -> {out_json_path}")
        except Exception as e:
            print(f"  [HPO] could not write json ({e})")

    return best
