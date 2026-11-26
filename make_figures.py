import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_json
from src.data_setup import CIFAR10_CLASSES


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def make_q1_figures(out_dir: str):
    q1 = load_json("logs/q1_results.json")
    q1a = q1["q1a"]
    q1b = q1["q1b"]

    # --- Fig 1: Q1(a) MLP Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    colors = {
        "MLP-Plain": "#1f77b4",
        "MLP-ResBN": "#d62728",
        "MLP-Reg": "#2ca02c",
        "MLP-ShiftAug": "#9467bd",
    }

    ax = axes[0]
    for name, res in q1a.items():
        epochs = np.arange(1, len(res["history"]["train_acc"]) + 1)
        ax.plot(epochs, np.array(res["history"]["train_acc"]) * 100,
                color=colors[name], linestyle="-", linewidth=1.6, label=f"{name} (Train)")
        ax.plot(epochs, np.array(res["history"]["val_acc"]) * 100,
                color=colors[name], linestyle="--", linewidth=1.6, label=f"{name} (Val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Training vs Validation Trajectories (~1.74M Params)")
    ax.legend(ncol=2, fontsize=6.8, frameon=True)

    ax = axes[1]
    names = list(q1a.keys())
    x = np.arange(len(names))
    w = 0.26
    tr_vals = [q1a[n]["final_train_acc"] * 100 for n in names]
    va_vals = [q1a[n]["best_val_acc"] * 100 for n in names]
    te_vals = [q1a[n]["test_acc"] * 100 for n in names]

    b1 = ax.bar(x - w, tr_vals, width=w, label="Final Train", color="#8da0cb")
    b2 = ax.bar(x, va_vals, width=w, label="Best Val", color="#fc8d62")
    b3 = ax.bar(x + w, te_vals, width=w, label="Test (Selected)", color="#66c2a5")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Final Train, Best Val, and Official Test Accuracy")
    ax.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_q1a_mlp_comparison.png"))
    plt.close(fig)

    # --- Fig 2: Q1(b) Memorisation Trajectory ---
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    h40 = q1b["noisy_40pct"]["history"]
    ep = np.arange(1, len(h40["val_acc"]) + 1)
    peak_ep = q1b["noisy_40pct"]["best_val_epoch"]

    ax = axes[0]
    ax.plot(ep, np.array(h40["train_clean_acc"]) * 100, color="#1f77b4", lw=1.8, label="Train Clean-Label Subset (60%)")
    ax.plot(ep, np.array(h40["train_noisy_acc"]) * 100, color="#d62728", lw=1.8, label="Train Wrong-Label Subset (40%)")
    ax.plot(ep, np.array(h40["val_acc"]) * 100, color="#2ca02c", lw=2.0, ls="--", label="Clean Validation Set")
    ax.axvline(peak_ep, color="black", ls=":", lw=1.2, label=f"Val Peak (Ep {peak_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) 40% Label-Noise Trajectory: Structure Before Memorisation")
    ax.legend(fontsize=7.0, loc="lower right")

    ax = axes[1]
    h0 = q1b["clean_0pct"]["history"]
    h100 = q1b["random_100pct"]["history"]
    ax.plot(ep, np.array(h0["val_acc"]) * 100, color="#1f77b4", lw=1.8, label="0% Noise (Val)")
    ax.plot(ep, np.array(h40["val_acc"]) * 100, color="#ff7f0e", lw=1.8, label="40% Noise (Val)")
    ax.plot(ep, np.array(h100["val_acc"]) * 100, color="#7f7f7f", lw=1.8, label="100% Random (Val)")
    ax.plot(ep, np.array(h100["train_acc"]) * 100, color="#d62728", lw=1.6, ls="--", label="100% Random (Train)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Capacity to Memorise Pure Noise vs Generalisation")
    ax.legend(fontsize=7.2, loc="center right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_q1b_memorisation.png"))
    plt.close(fig)


def make_q2_figures(out_dir: str):
    q2 = load_json("logs/q2_results.json")
    conds = ["clean", "pixel_permuted", "patch_8x8_permuted"]
    cond_labels = ["Clean", "Pixel-Permuted", "8x8 Patch-Permuted"]

    # --- Fig 3: Q2 Test Acc & Per-Class Breakdown ---
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    x = np.arange(len(conds))
    w = 0.34
    cnn_acc = [q2["conditions"][c]["FairCNN"]["test_acc"] * 100 for c in conds]
    mlp_acc = [q2["conditions"][c]["FairMLP"]["test_acc"] * 100 for c in conds]
    b1 = ax.bar(x - w / 2, cnn_acc, width=w, label="FairCNN (292k)", color="#1f77b4")
    b2 = ax.bar(x + w / 2, mlp_acc, width=w, label="FairMLP (291k)", color="#ff7f0e")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylim(0, 92)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("(a) Matched-Budget CNN vs MLP Across Spatial Conditions")
    ax.legend()

    ax = axes[1]
    clean_pc = [q2["conditions"]["clean"]["FairCNN"]["analysis"]["per_class_acc"][c] * 100 for c in CIFAR10_CLASSES]
    patch_pc = [q2["conditions"]["patch_8x8_permuted"]["FairCNN"]["analysis"]["per_class_acc"][c] * 100 for c in CIFAR10_CLASSES]
    cx = np.arange(10)
    w2 = 0.38
    ax.bar(cx - w2 / 2, clean_pc, width=w2, label="Clean CNN", color="#1f77b4")
    ax.bar(cx + w2 / 2, patch_pc, width=w2, label="8x8 Patch-Perm CNN", color="#2ca02c")
    ax.set_xticks(cx)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=32, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("(b) FairCNN Per-Class Accuracy: Clean vs 8x8 Patch-Permuted")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_q2_cnn_vs_mlp.png"))
    plt.close(fig)

    # --- Fig 4: Q2(c) Confidence Calibration & Error Structure ---
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect Calibration")
    for cond, lbl, col in [
        ("clean", "Clean FairCNN", "#1f77b4"),
        ("patch_8x8_permuted", "Patch-Perm FairCNN", "#2ca02c"),
        ("pixel_permuted", "Pixel-Perm FairCNN", "#d62728"),
    ]:
        bins = q2["conditions"][cond]["FairCNN"]["analysis"]["bin_stats"]
        confs = [b["conf"] for b in bins if b["count"] > 0]
        accs = [b["acc"] for b in bins if b["count"] > 0]
        ece = q2["conditions"][cond]["FairCNN"]["analysis"]["ece"]
        ax.plot(confs, accs, marker="o", ms=4, lw=1.6, color=col, label=f"{lbl} (ECE={ece:.3f})")
    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Empirical Test Accuracy")
    ax.set_title("(a) FairCNN Test Calibration Across Conditions")
    ax.legend(fontsize=7.3)

    ax = axes[1]
    labels_bar = ["Clean\nCNN", "Patch-Perm\nCNN", "Pixel-Perm\nCNN", "Clean\nMLP"]
    entries = [
        q2["conditions"]["clean"]["FairCNN"]["analysis"],
        q2["conditions"]["patch_8x8_permuted"]["FairCNN"]["analysis"],
        q2["conditions"]["pixel_permuted"]["FairCNN"]["analysis"],
        q2["conditions"]["clean"]["FairMLP"]["analysis"],
    ]
    c_corr = [e["mean_conf_correct"] * 100 for e in entries]
    c_wrong = [e["mean_conf_wrong"] * 100 for e in entries]
    bx = np.arange(len(labels_bar))
    ax.bar(bx - 0.18, c_corr, width=0.35, label="Correct Predictions", color="#4daf4a")
    ax.bar(bx + 0.18, c_wrong, width=0.35, label="Wrong Predictions", color="#e41a1c")
    for i, (vc, vw) in enumerate(zip(c_corr, c_wrong)):
        ax.text(i - 0.18, vc + 1.0, f"{vc:.1f}", ha="center", fontsize=7)
        ax.text(i + 0.18, vw + 1.0, f"{vw:.1f}", ha="center", fontsize=7)
    ax.set_xticks(bx)
    ax.set_xticklabels(labels_bar)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Mean Confidence (%)")
    ax.set_title("(b) Mean Softmax Confidence: Correct vs Incorrect")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_q2c_confidence_errors.png"))
    plt.close(fig)


def make_q3_figures(out_dir: str):
    q3 = load_json("logs/q3_results.json")
    samples = np.load("logs/q3_samples.npz")

    # --- Fig 5: Q3(a) Bottlenecks & Q3(b) Denoising Classification ---
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    dims = sorted(int(k) for k in q3["q3a"]["models"].keys())
    val_mses = [q3["q3a"]["models"][str(d)]["val_mse"] for d in dims]
    val_psnrs = [q3["q3a"]["models"][str(d)]["val_psnr"] for d in dims]
    x = np.arange(len(dims))
    ax.bar(x, val_mses, width=0.45, color=["#9ecae1", "#4292c6", "#08519c"])
    for i, (m, p) in enumerate(zip(val_mses, val_psnrs)):
        ax.text(i, m + 0.0005, f"{m:.4f}\n({p:.1f} dB)", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"z = {d}" for d in dims])
    ax.set_ylim(0, max(val_mses) * 1.25)
    ax.set_ylabel("Validation MSE")
    ax.set_title("(a) Q3(a) Bottleneck Size vs Val Reconstruction Error")

    ax = axes[1]
    cls_map = q3["q3b"]["test_classification"]
    keys = ["clean_images", "corrupted_images", "clean_ae_recons", "dae_recons"]
    lbls = ["Clean\nTest", "Noisy\n(σ=0.25)", "Clean-AE\nRecon", "Denoising-AE\nRecon"]
    accs = [cls_map[k]["acc"] * 100 for k in keys]
    cols = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(np.arange(len(keys)), accs, width=0.52, color=cols)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.2, f"{a:.1f}%", ha="center", fontsize=7.8)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(lbls)
    ax.set_ylim(0, 92)
    ax.set_ylabel("Fixed FairCNN Test Accuracy (%)")
    ax.set_title("(b) Q3(b) Classification Recovery via Denoising AE")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_q3ab_ae_denoising.png"))
    plt.close(fig)

    # --- Fig 6: Q3(b) Visual Reconstructions & Q3(c) Recon vs Linear Probe ---
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    # Left: 4x5 visual grid rendered onto axes[0] using composite image
    n_show = 5
    rows = [samples["clean"][:n_show], samples["noisy"][:n_show],
            samples["clean_ae_recon"][:n_show], samples["dae_recon"][:n_show]]
    row_names = ["Clean", "Noisy", "Clean-AE", "DAE"]
    pad = 2
    grid_h = 4 * 32 + 3 * pad
    grid_w = n_show * 32 + (n_show - 1) * pad
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.float32)
    for r_idx, r_arr in enumerate(rows):
        for c_idx in range(n_show):
            img = np.transpose(np.clip(r_arr[c_idx], 0, 1), (1, 2, 0))
            y0 = r_idx * (32 + pad)
            x0 = c_idx * (32 + pad)
            canvas[y0:y0 + 32, x0:x0 + 32] = img
    axes[0].imshow(canvas)
    axes[0].set_yticks([r * (32 + pad) + 16 for r in range(4)])
    axes[0].set_yticklabels(row_names)
    axes[0].set_xticks([c * (32 + pad) + 16 for c in range(n_show)])
    axes[0].set_xticklabels([CIFAR10_CLASSES[int(samples["labels"][c])] for c in range(n_show)], fontsize=7.5)
    axes[0].grid(False)
    axes[0].set_title("(a) Test Denoising Visual Comparison (z=256)")

    ax = axes[1]
    probes = q3["q3c"]["clean_image_probes"]
    names = list(probes.keys())
    psnrs = [probes[n]["test_clean_psnr"] for n in names]
    probe_accs = [probes[n]["probe"]["test_acc"] * 100 for n in names]
    cols_p = ["#9ecae1", "#4292c6", "#08519c", "#2ca02c"]
    for n, p, a, col in zip(names, psnrs, probe_accs, cols_p):
        ax.scatter(p, a, s=90, color=col, edgecolor="black", zorder=4)
        offset = (-36, -12) if n == "AE-256" else (6, 4)
        ax.annotate(n, (p, a), textcoords="offset points", xytext=offset, fontsize=8)
    ax.set_xlim(17.2, 24.4)
    ax.set_ylim(31.5, 51.0)
    ax.set_xlabel("Clean Test Reconstruction PSNR (dB)")
    ax.set_ylabel("Frozen Linear Probe Test Accuracy (%)")
    ax.set_title("(b) Q3(c) Reconstruction Quality vs Class Information")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_q3bc_visuals_probes.png"))
    plt.close(fig)


def make_q4_figures(out_dir: str):
    q4 = load_json("logs/q4_results.json")
    vis = np.load("logs/q4_visuals.npz")

    # --- Fig 7: Q4(a) SVM C and Kernel Sweep ---
    sweep = q4["q4a"]["sweep"]
    kernels = sorted(set(r["kernel"] for r in sweep))
    k_cols = {"linear": "#1f77b4", "rbf": "#d62728", "poly": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    for k in kernels:
        rows = sorted([r for r in sweep if r["kernel"] == k], key=lambda r: r["C"])
        cs = [r["C"] for r in rows]
        ax.semilogx(cs, [r["val_acc"] * 100 for r in rows], marker="o", lw=1.8, color=k_cols[k], label=f"{k} (Val)")
        ax.semilogx(cs, [r["train_acc"] * 100 for r in rows], ls="--", lw=1.2, alpha=0.65, color=k_cols[k], label=f"{k} (Train)")
    ax.set_xlabel("Regularisation Parameter C (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Q4(a) SVM Train & Validation Accuracy vs C")
    ax.legend(ncol=2, fontsize=6.8)

    ax = axes[1]
    for k in kernels:
        rows = sorted([r for r in sweep if r["kernel"] == k], key=lambda r: r["C"])
        cs = [r["C"] for r in rows]
        ax.semilogx(cs, [r["sv_fraction"] * 100 for r in rows], marker="s", lw=1.8, color=k_cols[k], label=k)
    ax.set_xlabel("Regularisation Parameter C (log scale)")
    ax.set_ylabel("Support Vector Fraction (%)")
    ax.set_title("(b) Q4(a) Support Vector Fraction vs C")
    ax.legend(loc="center left")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_q4a_svm_sweep.png"))
    plt.close(fig)

    # --- Fig 8: Q4(b) Support Vectors & Near-Boundary Test Images (Top 3 Difficult Pairs) ---
    pairs = q4["q4b"]["difficult_pairs"][:3]
    fig, axes = plt.subplots(3, 6, figsize=(9.2, 5.0))
    for r_idx, pinfo in enumerate(pairs):
        pkey = pinfo["pair"]
        sv_imgs = vis[f"{pkey}_sv_imgs"][:3]
        te_imgs = vis[f"{pkey}_test_imgs"][:3]
        for j in range(3):
            ax = axes[r_idx, j]
            ax.imshow(np.transpose(np.clip(sv_imgs[j], 0, 1), (1, 2, 0)))
            rec = pinfo["support_vectors"][j]
            ax.set_title(f"SV: {rec['true_label'][:4]}\n(s={rec['ovo_score']:+.3f})", fontsize=6.8)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(pkey.replace("_vs_", "\nvs "), fontsize=7.5)

            ax2 = axes[r_idx, j + 3]
            ax2.imshow(np.transpose(np.clip(te_imgs[j], 0, 1), (1, 2, 0)))
            trec = pinfo["near_boundary_test"][j]
            ax2.set_title(
                f"T:{trec['true_label'][:3]}->P:{trec['pred_label'][:3]}\n(s={trec['ovo_score']:+.3f})",
                fontsize=6.6,
                color="green" if trec["correct"] else "red",
            )
            ax2.set_xticks([]); ax2.set_yticks([])

    fig.suptitle("Q4(b) Difficult Class Pairs: Training Support Vectors (Cols 1-3) & Near-Boundary Test Images (Cols 4-6)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig8_q4b_boundary_examples.png"))
    plt.close(fig)

    # --- Fig 9: Q4(c) Contingency Table & SVM vs Clustering Disagreement ---
    sel_clus = q4["q4c"]["selected_test_eval"]
    cont = np.array(sel_clus["contingency_table"], dtype=np.float64)
    # Normalize each row or column to show class composition
    cont_norm = cont / np.maximum(cont.sum(axis=0, keepdims=True), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    im = ax.imshow(cont_norm * 100, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(np.arange(cont.shape[0]))
    ax.set_yticklabels([f"C{k}" for k in range(cont.shape[0])], fontsize=7.5)
    ax.set_xlabel("True CIFAR-10 Class")
    ax.set_ylabel(f"Cluster ({sel_clus['method']}, k={sel_clus['k']})")
    ax.set_title("(a) Test Contingency (% of Each Class per Cluster)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    summary = sel_clus["per_class_summary"]
    svm_accs = [summary[c]["svm_test_acc"] * 100 for c in CIFAR10_CLASSES]
    dom_fracs = [summary[c]["dominant_fraction"] * 100 for c in CIFAR10_CLASSES]
    cx = np.arange(10)
    w = 0.36
    ax.bar(cx - w / 2, svm_accs, width=w, label="SVM Test Acc (%)", color="#1f77b4")
    ax.bar(cx + w / 2, dom_fracs, width=w, label="Max Cluster Conc. (%)", color="#ff7f0e")
    ax.set_xticks(cx)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0, 85)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(b) Per-Class SVM Accuracy vs Cluster Concentration")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig9_q4c_clustering_vs_svm.png"))
    plt.close(fig)


def main():
    os.makedirs("figures", exist_ok=True)
    make_q1_figures("figures")
    make_q2_figures("figures")
    make_q3_figures("figures")
    make_q4_figures("figures")
    print("All figures generated in figures/")


if __name__ == "__main__":
    main()
