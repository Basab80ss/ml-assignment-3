import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.utils import set_seed, count_parameters, load_config, save_json, get_device
from src.data_setup import (
    CIFAR10_CLASSES,
    load_cifar10_splits,
    standardize_images,
    add_gaussian_noise,
    TensorImageDataset,
)
from src.models import ConvAutoencoder, FairCNN
from src.train_eval import train_autoencoder, evaluate_autoencoder, evaluate_classifier


def eval_cnn_on_01(cnn: FairCNN, x_01: np.ndarray, y: np.ndarray,
                   mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict:
    """Standardize [0, 1] images with training stats and evaluate the fixed Q2(a) CNN."""
    x_norm = standardize_images(x_01, mean, std)
    loader = DataLoader(TensorImageDataset(x_norm, y), batch_size=256, shuffle=False)
    res = evaluate_classifier(cnn, loader, device)
    per_class = {}
    for idx, cname in enumerate(CIFAR10_CLASSES):
        mask = res["targets"] == idx
        per_class[cname] = float((res["preds"][mask] == res["targets"][mask]).mean())
    return {
        "acc": res["acc"],
        "loss": res["loss"],
        "per_class_acc": per_class,
        "mean_conf": float(res["probs"].max(axis=1).mean()),
    }


def run_linear_probe(z_tr: np.ndarray, y_tr: np.ndarray,
                     z_va: np.ndarray, y_va: np.ndarray,
                     z_te: np.ndarray, y_te: np.ndarray,
                     c_grid: list, seed: int) -> dict:
    """Fit StandardScaler + LogisticRegression on frozen latent codes, tuning C on val only."""
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(z_tr)
    x_va = scaler.transform(z_va)
    x_te = scaler.transform(z_te)

    best_c = None
    best_val_acc = -1.0
    best_clf = None
    sweep = {}

    for c_val in c_grid:
        clf = LogisticRegression(C=c_val, max_iter=1000, random_state=seed)
        clf.fit(x_tr, y_tr)
        tr_acc = float(clf.score(x_tr, y_tr))
        va_acc = float(clf.score(x_va, y_va))
        sweep[str(c_val)] = {"train_acc": tr_acc, "val_acc": va_acc}
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_c = c_val
            best_clf = clf

    test_acc = float(best_clf.score(x_te, y_te))
    return {
        "best_C": best_c,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "sweep": sweep,
    }


def main():
    cfg = load_config()
    seed = cfg["student"]["seed"]
    bs = cfg["q3"]["batch_size"]
    epochs = cfg["q3"]["epochs"]
    lr = cfg["q3"]["lr"]
    wd = cfg["q3"]["weight_decay"]
    bottlenecks = cfg["q3"]["bottlenecks"]
    noise_std = cfg["q3"]["noise_std"]
    c_grid = cfg["q3"]["probe_C_grid"]
    device = get_device()
    print(f"Using device: {device}")

    data = load_cifar10_splits(cfg)
    x_tr_01, y_tr = data["x_train_01"], data["y_train"]
    x_va_01, y_va = data["x_val_01"], data["y_val"]
    x_te_01, y_te = data["x_test_01"], data["y_test"]

    os.makedirs("checkpoints", exist_ok=True)
    q3a_results = {}
    clean_models = {}

    # --- Q3(a): Three bottleneck capacities ---
    print("\n=== Q3(a): Training Convolutional Autoencoders Across Bottlenecks ===")
    for idx, dim in enumerate(bottlenecks):
        print(f"\nTraining ConvAE (latent_dim={dim})...")
        set_seed(seed + 300 + idx)
        ae = ConvAutoencoder(latent_dim=dim)
        n_params = count_parameters(ae)
        print(f"  Parameters: {n_params:,}")

        history, best_state, best_epoch, best_val_mse = train_autoencoder(
            model=ae,
            x_train_01=x_tr_01,
            x_val_in_01=x_va_01,
            x_val_target_01=x_va_01,
            epochs=epochs,
            batch_size=bs,
            lr=lr,
            weight_decay=wd,
            device=device,
            noise_std=0.0,
            seed=seed + 300 + idx,
        )

        ae.load_state_dict(best_state)
        torch.save(best_state, f"checkpoints/q3a_ae_{dim}.pt")
        clean_models[dim] = ae

        val_eval = evaluate_autoencoder(ae, x_va_01, x_va_01, bs, device)
        print(f"  -> Best Val MSE: {best_val_mse:.5f} (PSNR {val_eval['psnr']:.2f} dB, ep {best_epoch})")

        q3a_results[str(dim)] = {
            "latent_dim": dim,
            "params": n_params,
            "best_epoch": best_epoch,
            "val_mse": val_eval["mse"],
            "val_psnr": val_eval["psnr"],
            "history": history,
        }

    # Select best clean AE on validation MSE only, then evaluate once on test
    selected_dim = min(bottlenecks, key=lambda d: q3a_results[str(d)]["val_mse"])
    selected_ae = clean_models[selected_dim]
    test_ae_eval = evaluate_autoencoder(selected_ae, x_te_01, x_te_01, bs, device)
    print(
        f"\nSelected Clean AE (val MSE criterion): latent_dim={selected_dim} | "
        f"Test MSE={test_ae_eval['mse']:.5f}, Test PSNR={test_ae_eval['psnr']:.2f} dB"
    )

    # --- Q3(b): Denoising Autoencoder & Classification Recovery ---
    print(f"\n=== Q3(b): Denoising Autoencoder (Gaussian noise sigma={noise_std}) ===")
    x_va_noisy = add_gaussian_noise(x_va_01, sigma=noise_std, seed=seed + 400)
    x_te_noisy = add_gaussian_noise(x_te_01, sigma=noise_std, seed=seed + 401)

    set_seed(seed + 350)
    dae = ConvAutoencoder(latent_dim=selected_dim)
    dae_params = count_parameters(dae)
    dae_hist, dae_best_state, dae_best_ep, dae_best_val_mse = train_autoencoder(
        model=dae,
        x_train_01=x_tr_01,
        x_val_in_01=x_va_noisy,
        x_val_target_01=x_va_01,
        epochs=epochs,
        batch_size=bs,
        lr=lr,
        weight_decay=wd,
        device=device,
        noise_std=noise_std,
        seed=seed + 350,
    )
    dae.load_state_dict(dae_best_state)
    torch.save(dae_best_state, f"checkpoints/q3b_dae_{selected_dim}.pt")

    # Reconstruct noisy test images with Clean AE baseline and Denoising AE
    clean_ae_on_noisy_test = evaluate_autoencoder(selected_ae, x_te_noisy, x_te_01, bs, device)
    dae_on_noisy_test = evaluate_autoencoder(dae, x_te_noisy, x_te_01, bs, device)
    dae_on_clean_test = evaluate_autoencoder(dae, x_te_01, x_te_01, bs, device)

    # Load fixed FairCNN trained in Q2(a)
    cnn = FairCNN(num_classes=10).to(device)
    cnn.load_state_dict(torch.load("checkpoints/q2a_fair_cnn.pt", map_location=device))

    q3b_cls = {
        "clean_images": eval_cnn_on_01(cnn, x_te_01, y_te, data["mean"], data["std"], device),
        "corrupted_images": eval_cnn_on_01(cnn, x_te_noisy, y_te, data["mean"], data["std"], device),
        "clean_ae_recons": eval_cnn_on_01(
            cnn, clean_ae_on_noisy_test["recons"], y_te, data["mean"], data["std"], device
        ),
        "dae_recons": eval_cnn_on_01(
            cnn, dae_on_noisy_test["recons"], y_te, data["mean"], data["std"], device
        ),
    }
    print("  Test Classification via Fixed Q2(a) FairCNN:")
    for k, v in q3b_cls.items():
        print(f"    {k:20s}: Acc = {v['acc']:.4f} | Mean Conf = {v['mean_conf']:.3f}")

    # Save 8 test sample reconstructions for report figures
    np.savez(
        "logs/q3_samples.npz",
        clean=x_te_01[:8],
        noisy=x_te_noisy[:8],
        clean_ae_recon=clean_ae_on_noisy_test["recons"][:8],
        dae_recon=dae_on_noisy_test["recons"][:8],
        labels=y_te[:8],
    )

    # --- Q3(c): Class Information (Linear Probe) vs Reconstruction Quality ---
    print("\n=== Q3(c): Frozen Representation Linear Probe vs Reconstruction Quality ===")
    encoders = {f"AE-{d}": clean_models[d] for d in bottlenecks}
    encoders[f"DAE-{selected_dim}"] = dae

    q3c_results = {}
    for name, enc_model in encoders.items():
        tr_eval = evaluate_autoencoder(enc_model, x_tr_01, x_tr_01, bs, device)
        va_eval = evaluate_autoencoder(enc_model, x_va_01, x_va_01, bs, device)
        te_eval = evaluate_autoencoder(enc_model, x_te_01, x_te_01, bs, device)

        probe = run_linear_probe(
            tr_eval["latents"], y_tr,
            va_eval["latents"], y_va,
            te_eval["latents"], y_te,
            c_grid=c_grid,
            seed=seed,
        )
        print(
            f"  {name:8s} | Clean Test MSE: {te_eval['mse']:.5f} (PSNR {te_eval['psnr']:.2f} dB) | "
            f"Linear Probe Val: {probe['best_val_acc']:.4f} -> Test: {probe['test_acc']:.4f} (C={probe['best_C']})"
        )
        q3c_results[name] = {
            "latent_dim": enc_model.latent_dim,
            "test_clean_mse": te_eval["mse"],
            "test_clean_psnr": te_eval["psnr"],
            "probe": probe,
        }

    # Also test DAE linear probe on corrupted test images to see robustness of learned code
    tr_noisy_eval = evaluate_autoencoder(
        dae, add_gaussian_noise(x_tr_01, sigma=noise_std, seed=seed + 402), x_tr_01, bs, device
    )
    va_noisy_eval = evaluate_autoencoder(dae, x_va_noisy, x_va_01, bs, device)
    probe_dae_noisy = run_linear_probe(
        tr_noisy_eval["latents"], y_tr,
        va_noisy_eval["latents"], y_va,
        dae_on_noisy_test["latents"], y_te,
        c_grid=c_grid,
        seed=seed,
    )

    out = {
        "q3a": {
            "models": q3a_results,
            "selected_dim": selected_dim,
            "selected_test_mse": test_ae_eval["mse"],
            "selected_test_psnr": test_ae_eval["psnr"],
        },
        "q3b": {
            "noise_std": noise_std,
            "dae_params": dae_params,
            "dae_best_epoch": dae_best_ep,
            "dae_best_val_mse": dae_best_val_mse,
            "dae_history": dae_hist,
            "test_recon_clean_ae_on_noisy": {
                "mse": clean_ae_on_noisy_test["mse"],
                "psnr": clean_ae_on_noisy_test["psnr"],
            },
            "test_recon_dae_on_noisy": {
                "mse": dae_on_noisy_test["mse"],
                "psnr": dae_on_noisy_test["psnr"],
            },
            "test_recon_dae_on_clean": {
                "mse": dae_on_clean_test["mse"],
                "psnr": dae_on_clean_test["psnr"],
            },
            "test_classification": q3b_cls,
        },
        "q3c": {
            "clean_image_probes": q3c_results,
            "dae_noisy_image_probe": {
                "test_mse_vs_clean": dae_on_noisy_test["mse"],
                "probe": probe_dae_noisy,
            },
        },
    }
    save_json(out, "logs/q3_results.json")
    print("\nSaved Q3 results to logs/q3_results.json")


if __name__ == "__main__":
    main()
