import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, count_parameters, load_config, save_json, get_device
from src.data_setup import load_cifar10_splits, standardize_images, TensorImageDataset
from src.models import MLPPlain, MLPResBN, MLPMem
from src.train_eval import train_classifier, evaluate_classifier


def run_q1a(cfg, data, device):
    print("\n=== Q1(a): Separating Reasons for Poor MLP Performance on CIFAR-10 ===")
    seed = cfg["student"]["seed"]
    bs = cfg["q1"]["batch_size"]
    epochs = cfg["q1"]["epochs_a"]
    lr = cfg["q1"]["lr_a"]

    x_tr = standardize_images(data["x_train_01"], data["mean"], data["std"])
    x_va = standardize_images(data["x_val_01"], data["mean"], data["std"])
    x_te = standardize_images(data["x_test_01"], data["mean"], data["std"])
    y_tr, y_va, y_te = data["y_train"], data["y_val"], data["y_test"]

    val_loader = DataLoader(TensorImageDataset(x_va, y_va), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorImageDataset(x_te, y_te), batch_size=256, shuffle=False)

    specs = [
        {
            "name": "MLP-Plain",
            "desc": "3072-512-256-128-10, ReLU, wd=1e-4, no aug",
            "builder": lambda: MLPPlain(dropout=0.0),
            "wd": cfg["q1"]["weight_decay_plain"],
            "shift_aug": False,
        },
        {
            "name": "MLP-ResBN",
            "desc": "3072-440 + 2xResBN(440)-10, ReLU+BN, wd=1e-4, no aug",
            "builder": lambda: MLPResBN(width=440),
            "wd": cfg["q1"]["weight_decay_plain"],
            "shift_aug": False,
        },
        {
            "name": "MLP-Reg",
            "desc": "3072-512-256-128-10, ReLU, Dropout=0.35, wd=1e-2, no aug",
            "builder": lambda: MLPPlain(dropout=cfg["q1"]["dropout_reg"]),
            "wd": cfg["q1"]["weight_decay_reg"],
            "shift_aug": False,
        },
        {
            "name": "MLP-ShiftAug",
            "desc": "3072-512-256-128-10, ReLU, wd=1e-4, RandomCrop(32,pad=4)+HFlip",
            "builder": lambda: MLPPlain(dropout=0.0),
            "wd": cfg["q1"]["weight_decay_plain"],
            "shift_aug": True,
        },
    ]

    results = {}
    for idx, spec in enumerate(specs):
        print(f"\nTraining {spec['name']} ({spec['desc']})...")
        set_seed(seed + idx)
        model = spec["builder"]()
        n_params = count_parameters(model)
        print(f"  Trainable parameters: {n_params:,}")

        g = torch.Generator().manual_seed(seed + idx)
        train_ds = TensorImageDataset(x_tr, y_tr, shift_augment=spec["shift_aug"])
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g)

        history, best_state, best_epoch, best_val_acc = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=spec["wd"],
            device=device,
            use_cosine=True,
        )

        # Evaluate selected (best validation) model once on the official test set
        model.load_state_dict(best_state)
        test_res = evaluate_classifier(model, test_loader, device)
        print(
            f"  -> Best Val Acc: {best_val_acc:.4f} (ep {best_epoch}) | "
            f"Final Test Acc: {test_res['acc']:.4f}"
        )

        results[spec["name"]] = {
            "description": spec["desc"],
            "params": n_params,
            "weight_decay": spec["wd"],
            "shift_aug": spec["shift_aug"],
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
            "test_acc": test_res["acc"],
            "test_loss": test_res["loss"],
            "history": history,
        }

    return results


def run_q1b(cfg, data, device):
    print("\n=== Q1(b): Memorisation vs Useful Structure Trajectory ===")
    seed = cfg["student"]["seed"]
    bs = cfg["q1"]["batch_size"]
    epochs = cfg["q1"]["epochs_b"]
    lr = cfg["q1"]["lr_b"]
    noise_rate = cfg["q1"]["noise_rate_b"]

    x_tr = standardize_images(data["x_train_01"], data["mean"], data["std"])
    x_va = standardize_images(data["x_val_01"], data["mean"], data["std"])
    x_te = standardize_images(data["x_test_01"], data["mean"], data["std"])
    y_tr_clean, y_va, y_te = data["y_train"], data["y_val"], data["y_test"]

    val_loader = DataLoader(TensorImageDataset(x_va, y_va), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorImageDataset(x_te, y_te), batch_size=256, shuffle=False)

    rng = np.random.default_rng(seed)
    n_tr = len(y_tr_clean)

    # Condition 1: 0% noise (clean)
    # Condition 2: 40% symmetric wrong labels (strictly wrong class so memorisation is unambiguous)
    y_tr_40 = y_tr_clean.copy()
    noisy_idx_40 = rng.choice(n_tr, size=int(noise_rate * n_tr), replace=False)
    noisy_mask_40 = np.zeros(n_tr, dtype=bool)
    noisy_mask_40[noisy_idx_40] = True
    for i in noisy_idx_40:
        choices = [c for c in range(10) if c != y_tr_clean[i]]
        y_tr_40[i] = int(rng.choice(choices))

    # Condition 3: 100% uniform random labels
    y_tr_100 = rng.integers(0, 10, size=n_tr, dtype=np.int64)
    noisy_mask_100 = np.ones(n_tr, dtype=bool)

    settings = [
        ("clean_0pct", y_tr_clean, np.zeros(n_tr, dtype=bool)),
        ("noisy_40pct", y_tr_40, noisy_mask_40),
        ("random_100pct", y_tr_100, noisy_mask_100),
    ]

    results = {}
    for idx, (label, y_train_cur, mask) in enumerate(settings):
        print(f"\nRunning Q1(b) setting: {label}...")
        set_seed(seed + 100 + idx)
        model = MLPMem()
        n_params = count_parameters(model)
        print(f"  MLPMem parameters: {n_params:,}")

        g = torch.Generator().manual_seed(seed + 100 + idx)
        train_ds = TensorImageDataset(x_tr, y_train_cur, shift_augment=False)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g)
        eval_train_loader = DataLoader(train_ds, batch_size=512, shuffle=False)

        # Constant lr (use_cosine=False) keeps memorisation pressure steady across epochs
        history, best_state, best_epoch, best_val_acc = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=0.0,
            device=device,
            use_cosine=False,
            clean_train_eval_loader=eval_train_loader,
            noisy_mask=mask,
        )

        # Final-epoch test accuracy and best-val test accuracy
        final_test_res = evaluate_classifier(model, test_loader, device)
        model.load_state_dict(best_state)
        best_val_test_res = evaluate_classifier(model, test_loader, device)

        print(
            f"  -> Peak Val Acc: {best_val_acc:.4f} (ep {best_epoch}, Test {best_val_test_res['acc']:.4f}) | "
            f"Final Ep Val Acc: {history['val_acc'][-1]:.4f} (Test {final_test_res['acc']:.4f})"
        )

        results[label] = {
            "params": n_params,
            "noise_fraction": float(mask.mean()),
            "best_val_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_test_acc": best_val_test_res["acc"],
            "final_epoch_val_acc": history["val_acc"][-1],
            "final_epoch_test_acc": final_test_res["acc"],
            "history": history,
        }

    return results


def main():
    cfg = load_config()
    device = get_device()
    print(f"Using device: {device}")
    data = load_cifar10_splits(cfg)

    q1a_res = run_q1a(cfg, data, device)
    q1b_res = run_q1b(cfg, data, device)

    save_json({"q1a": q1a_res, "q1b": q1b_res}, "logs/q1_results.json")
    print("\nSaved Q1 results to logs/q1_results.json")


if __name__ == "__main__":
    main()
