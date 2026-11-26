import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from src.utils import set_seed, count_parameters, load_config, save_json, get_device
from src.data_setup import (
    CIFAR10_CLASSES,
    load_cifar10_splits,
    standardize_images,
    permute_pixels,
    permute_patches,
    TensorImageDataset,
)
from src.models import FairCNN, FairMLP
from src.train_eval import train_classifier, evaluate_classifier


def compute_error_and_confidence(res: dict, n_bins: int = 10) -> dict:
    """Compute per-class accuracy, confusion matrix, confidence stats, and ECE."""
    preds = res["preds"]
    probs = res["probs"]
    targets = res["targets"]
    confs = probs.max(axis=1)
    correct = preds == targets

    per_class_acc = {}
    for c_idx, c_name in enumerate(CIFAR10_CLASSES):
        mask = targets == c_idx
        per_class_acc[c_name] = float((preds[mask] == targets[mask]).mean())

    cm = confusion_matrix(targets, preds, labels=list(range(10)))

    # Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            in_bin = (confs >= lo) & (confs <= hi)
        else:
            in_bin = (confs >= lo) & (confs < hi)
        count = int(in_bin.sum())
        if count > 0:
            acc_bin = float(correct[in_bin].mean())
            conf_bin = float(confs[in_bin].mean())
            ece += (count / len(targets)) * abs(acc_bin - conf_bin)
        else:
            acc_bin, conf_bin = 0.0, 0.0
        bin_stats.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": count,
                          "acc": acc_bin, "conf": conf_bin})

    high_conf_err = float((~correct & (confs >= 0.8)).sum() / max((~correct).sum(), 1))
    top_confusions = []
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    flat_order = np.argsort(cm_no_diag.ravel())[::-1][:6]
    for flat_i in flat_order:
        r, c = divmod(int(flat_i), 10)
        top_confusions.append({
            "true": CIFAR10_CLASSES[r],
            "pred": CIFAR10_CLASSES[c],
            "count": int(cm[r, c]),
        })

    return {
        "overall_acc": float(correct.mean()),
        "mean_conf_all": float(confs.mean()),
        "mean_conf_correct": float(confs[correct].mean()) if correct.any() else 0.0,
        "mean_conf_wrong": float(confs[~correct].mean()) if (~correct).any() else 0.0,
        "high_conf_wrong_frac": high_conf_err,
        "ece": float(ece),
        "bin_stats": bin_stats,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions,
    }


def main():
    cfg = load_config()
    seed = cfg["student"]["seed"]
    bs = cfg["q2"]["batch_size"]
    epochs = cfg["q2"]["epochs"]
    lr = cfg["q2"]["lr"]
    wd = cfg["q2"]["weight_decay"]
    patch_size = cfg["q2"]["patch_size"]
    device = get_device()
    print(f"Using device: {device}")

    data = load_cifar10_splits(cfg)
    x_tr_clean = standardize_images(data["x_train_01"], data["mean"], data["std"])
    x_va_clean = standardize_images(data["x_val_01"], data["mean"], data["std"])
    x_te_clean = standardize_images(data["x_test_01"], data["mean"], data["std"])
    y_tr, y_va, y_te = data["y_train"], data["y_val"], data["y_test"]

    # Apply identical spatial transformations to train, val, and test
    conditions = {
        "clean": (x_tr_clean, x_va_clean, x_te_clean),
        "pixel_permuted": (
            permute_pixels(x_tr_clean, seed=seed),
            permute_pixels(x_va_clean, seed=seed),
            permute_pixels(x_te_clean, seed=seed),
        ),
        "patch_8x8_permuted": (
            permute_patches(x_tr_clean, patch_size=patch_size, seed=seed + 1),
            permute_patches(x_va_clean, patch_size=patch_size, seed=seed + 1),
            permute_patches(x_te_clean, patch_size=patch_size, seed=seed + 1),
        ),
    }

    os.makedirs("checkpoints", exist_ok=True)
    results = {"models": {}, "conditions": {}}

    for cond_name, (x_tr, x_va, x_te) in conditions.items():
        print(f"\n=== Q2 Condition: {cond_name} ===")
        val_loader = DataLoader(TensorImageDataset(x_va, y_va), batch_size=256, shuffle=False)
        test_loader = DataLoader(TensorImageDataset(x_te, y_te), batch_size=256, shuffle=False)

        cond_res = {}
        for m_idx, (m_name, builder) in enumerate([
            ("FairCNN", lambda: FairCNN(num_classes=10)),
            ("FairMLP", lambda: FairMLP(width=66, num_classes=10)),
        ]):
            print(f"\nTraining {m_name} on {cond_name}...")
            set_seed(seed + 200 + m_idx)
            model = builder()
            n_params = count_parameters(model)
            results["models"][m_name] = {"params": n_params}
            print(f"  Parameters: {n_params:,}")

            g = torch.Generator().manual_seed(seed + 200 + m_idx)
            train_loader = DataLoader(
                TensorImageDataset(x_tr, y_tr), batch_size=bs, shuffle=True, generator=g
            )

            history, best_state, best_epoch, best_val_acc = train_classifier(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                weight_decay=wd,
                device=device,
                use_cosine=True,
            )

            model.load_state_dict(best_state)
            if cond_name == "clean" and m_name == "FairCNN":
                torch.save(best_state, "checkpoints/q2a_fair_cnn.pt")

            test_eval = evaluate_classifier(model, test_loader, device)
            analysis = compute_error_and_confidence(test_eval)
            print(
                f"  -> Best Val Acc: {best_val_acc:.4f} (ep {best_epoch}) | "
                f"Test Acc: {analysis['overall_acc']:.4f} | "
                f"Conf(corr/wrong): {analysis['mean_conf_correct']:.3f}/{analysis['mean_conf_wrong']:.3f} | "
                f"ECE: {analysis['ece']:.4f}"
            )

            cond_res[m_name] = {
                "params": n_params,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "final_train_acc": history["train_acc"][-1],
                "test_loss": test_eval["loss"],
                "test_acc": analysis["overall_acc"],
                "analysis": analysis,
                "history": history,
            }

        results["conditions"][cond_name] = cond_res

    save_json(results, "logs/q2_results.json")
    print("\nSaved Q2 results to logs/q2_results.json")


if __name__ == "__main__":
    main()
