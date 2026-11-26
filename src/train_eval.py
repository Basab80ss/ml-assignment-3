import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def evaluate_classifier(model: nn.Module, loader: DataLoader, device: torch.device):
    """Evaluate cross-entropy loss, accuracy, predictions, probabilities, and targets."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item() * len(yb)
            total_samples += len(yb)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    acc = float((preds == targets).mean())
    return {
        "loss": total_loss / total_samples,
        "acc": acc,
        "preds": preds,
        "probs": probs,
        "targets": targets,
    }


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    use_cosine: bool = True,
    clean_train_eval_loader: DataLoader = None,
    noisy_mask: np.ndarray = None,
):
    """Train a classifier and track per-epoch train/val history, keeping best val checkpoint."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_cosine
        else None
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    if noisy_mask is not None:
        history["train_clean_acc"] = []
        history["train_noisy_acc"] = []
        history["train_clean_loss"] = []
        history["train_noisy_loss"] = []
        history["weight_norm"] = []

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        val_res = evaluate_classifier(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])

        if noisy_mask is not None and clean_train_eval_loader is not None:
            # Evaluate clean-label vs corrupted-label training subsets on the current model
            model.eval()
            losses_list, correct_list = [], []
            ce_none = nn.CrossEntropyLoss(reduction="none")
            with torch.no_grad():
                for xb, yb in clean_train_eval_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    losses_list.append(ce_none(logits, yb).cpu().numpy())
                    correct_list.append((logits.argmax(dim=1) == yb).cpu().numpy())
            per_sample_loss = np.concatenate(losses_list)
            per_sample_corr = np.concatenate(correct_list)
            clean_mask = ~noisy_mask
            history["train_clean_acc"].append(float(per_sample_corr[clean_mask].mean()))
            history["train_clean_loss"].append(float(per_sample_loss[clean_mask].mean()))
            if noisy_mask.any():
                history["train_noisy_acc"].append(float(per_sample_corr[noisy_mask].mean()))
                history["train_noisy_loss"].append(float(per_sample_loss[noisy_mask].mean()))
            else:
                history["train_noisy_acc"].append(float("nan"))
                history["train_noisy_loss"].append(float("nan"))
            wnorm = float(
                np.sqrt(sum(p.detach().pow(2).sum().item() for p in model.parameters()))
            )
            history["weight_norm"].append(wnorm)

        if val_res["acc"] > best_val_acc:
            best_val_acc = val_res["acc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:02d}/{epochs} | "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                f"Val Loss {val_res['loss']:.4f} Acc {val_res['acc']:.4f} | {elapsed:.1f}s"
            )

    return history, best_state, best_epoch, best_val_acc


def evaluate_autoencoder(model: nn.Module, x_in_01: np.ndarray, x_target_01: np.ndarray,
                         batch_size: int, device: torch.device):
    """Evaluate reconstruction MSE, PSNR, and produce reconstructed images and latent codes."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_in_01), torch.from_numpy(x_target_01)),
        batch_size=batch_size,
        shuffle=False,
    )
    total_mse = 0.0
    total_elems = 0
    recons, latents = [], []
    with torch.no_grad():
        for xb, tb in loader:
            xb, tb = xb.to(device), tb.to(device)
            recon, z = model(xb)
            diff = (recon - tb).pow(2)
            total_mse += diff.sum().item()
            total_elems += diff.numel()
            recons.append(recon.cpu().numpy())
            latents.append(z.cpu().numpy())

    mse = total_mse / total_elems
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    return {
        "mse": float(mse),
        "psnr": psnr,
        "recons": np.concatenate(recons, axis=0),
        "latents": np.concatenate(latents, axis=0),
    }


def train_autoencoder(
    model: nn.Module,
    x_train_01: np.ndarray,
    x_val_in_01: np.ndarray,
    x_val_target_01: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    noise_std: float = 0.0,
    seed: int = 26002,
):
    """Train clean or denoising convolutional autoencoder using MSE on [0, 1] images."""
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_01)),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )

    history = {"train_mse": [], "val_mse": [], "val_psnr": []}
    best_val_mse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_mse = 0.0
        total_samples = 0

        for (clean_b,) in train_loader:
            clean_b = clean_b.to(device)
            if noise_std > 0.0:
                noisy_b = torch.clamp(clean_b + noise_std * torch.randn_like(clean_b), 0.0, 1.0)
            else:
                noisy_b = clean_b

            optimizer.zero_grad()
            recon, _ = model(noisy_b)
            loss = criterion(recon, clean_b)
            loss.backward()
            optimizer.step()

            running_mse += loss.item() * len(clean_b)
            total_samples += len(clean_b)

        scheduler.step()
        train_mse = running_mse / total_samples
        val_res = evaluate_autoencoder(model, x_val_in_01, x_val_target_01, batch_size, device)

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_res["mse"])
        history["val_psnr"].append(val_res["psnr"])

        if val_res["mse"] < best_val_mse:
            best_val_mse = val_res["mse"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:02d}/{epochs} | "
                f"Train MSE {train_mse:.5f} | "
                f"Val MSE {val_res['mse']:.5f} (PSNR {val_res['psnr']:.2f} dB) | {elapsed:.1f}s"
            )

    return history, best_state, best_epoch, best_val_mse
