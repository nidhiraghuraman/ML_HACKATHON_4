"""Phase 4: Bayesian hyperparameter optimisation of the Phase 3 selected model.

Method: Optuna's Tree-structured Parzen Estimator (TPE). After a few random trials, TPE splits the
trials seen so far into "good" and "bad" groups by score, models each group's hyperparameter
values with a probability density, and proposes the next values where good/bad density is
highest. Every new trial uses what the previous trials revealed (sequential model-based, i.e.
Bayesian, optimisation), unlike grid or random search.

Objective: mean Top-1 accuracy over trip-grouped folds of a reproducible subset of whole GPS points
from all training trips (student-test data is never touched). Unpromising trials are stopped
early with the median pruner.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from . import modeling as M
from . import metrics as Q


def suggest_params(trial, space: dict, fixed: dict) -> dict:
    """Concrete parameters for one trial: searched values from `space`, the rest from `fixed`.
    space example: {"learning_rate": {"type": "float", "low": 0.02, "high": 0.3, "log": True}}"""
    params = dict(fixed)
    for name, s in space.items():
        t = s["type"]
        if t == "int":
            params[name] = trial.suggest_int(name, int(s["low"]), int(s["high"]), step=int(s.get("step", 1)), log=bool(s.get("log", False)))
        elif t == "float":
            params[name] = trial.suggest_float(name, float(s["low"]), float(s["high"]), log=bool(s.get("log", False)))
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, list(s["choices"]))
        else:
            raise ValueError(f"unknown search-space type {t} for {name}")
    return params


def sample_observation_fraction(cand: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Keep a random `fraction` of GPS points (with ALL their candidate rows) from every trip, so
    within-point ranking stays valid and every trip stays represented."""
    if fraction >= 1.0:
        return cand
    keys = cand[M.OBS_KEY].drop_duplicates()
    keys = keys.groupby("trip_id", group_keys=False).sample(frac=fraction, random_state=seed)
    return cand.merge(keys, on=M.OBS_KEY, how="inner").reset_index(drop=True)


def cv_score(model_name: str, params: dict, data: pd.DataFrame, feats: list[str], folds: list[dict], seed: int,
             trial=None) -> dict:
    """Fit/evaluate one parameter set on trip-grouped folds of `data` (needs in_fit / in_rank flags).
    Reports the running mean Top-1 to Optuna after each fold (for pruning)."""
    tops, mrrs, f1s, lls, fits = [], [], [], [], []
    for k, f in enumerate(folds):
        M.assert_disjoint(f["train_trips"], f["val_trips"])
        tr = data[data.in_fit & data.trip_id.isin(f["train_trips"])]
        va = data[data.trip_id.isin(f["val_trips"]) & (data.in_fit | data.in_rank)]
        model = M.build_model(model_name, params, seed)
        t0 = time.perf_counter()
        model.fit(tr[feats].to_numpy(np.float32), tr.label.to_numpy())
        fits.append(time.perf_counter() - t0)
        p = model.predict_proba(va[feats].to_numpy(np.float32))[:, 1]
        rk = va.in_rank.to_numpy()
        rm, _ = Q.ranking_metrics(va[rk], p[rk])
        fm = va.in_fit.to_numpy()
        y = va.label.to_numpy()[fm]
        cm = Q.candidate_metrics(y, p[fm], (p[fm] >= 0.5).astype(int), True)
        tops.append(rm["top1"]); mrrs.append(rm["mrr"]); f1s.append(cm["macro_f1"]); lls.append(cm["log_loss"])
        if trial is not None:
            trial.report(float(np.mean(tops)), k)
            if trial.should_prune():
                import optuna
                trial.set_user_attr("folds_completed", k + 1)
                raise optuna.TrialPruned()
    res = {"top1": float(np.mean(tops)), "top1_std": float(np.std(tops, ddof=1)) if len(tops) > 1 else 0.0,
           "mrr": float(np.mean(mrrs)), "macro_f1": float(np.mean(f1s)), "log_loss": float(np.mean(lls)),
           "fit_s": float(np.mean(fits)), "fold_top1": [float(x) for x in tops]}
    if trial is not None:
        for k_, v in res.items():
            trial.set_user_attr(k_, v)
        trial.set_user_attr("folds_completed", len(folds))
    return res


def run_study(model_name: str, space: dict, fixed: dict, data: pd.DataFrame, feats: list[str], folds: list[dict],
              n_trials: int, seed: int, timeout_s: float | None = None, startup_trials: int = 5, log=print):
    """Run the TPE search. Trial 0 is always the untuned Phase 3 configuration, so the search
    starts from, and is compared against, the baseline settings."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=startup_trials, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=f"phase4_{model_name}")
    study.enqueue_trial({k: fixed[k] for k in space if k in fixed and fixed[k] is not None})

    def objective(trial):
        params = suggest_params(trial, space, fixed)
        t0 = time.perf_counter()
        res = cv_score(model_name, params, data, feats, folds, seed, trial)
        trial.set_user_attr("trial_s", time.perf_counter() - t0)
        return res["top1"]

    def cb(study_, trial):
        val = f"{trial.value:.4f}" if trial.value is not None else "  -   "
        best = f"{study_.best_value:.4f}" if any(t.state.name == "COMPLETE" for t in study_.trials) else "-"
        shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in trial.params.items()}
        log(f"    trial {trial.number:>3} {trial.state.name:<8} top1={val} best={best} params={shown}")

    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, callbacks=[cb], gc_after_trial=True)
    return study


def trials_table(study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        r = {"trial": t.number, "state": t.state.name, "value_top1": t.value,
             "duration_s": t.duration.total_seconds() if t.duration else np.nan}
        r.update({f"param_{k}": v for k, v in t.params.items()})
        r.update({k: v for k, v in t.user_attrs.items() if k != "fold_top1"})
        rows.append(r)
    return pd.DataFrame(rows)
