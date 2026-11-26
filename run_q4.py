import os
import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)

from src.utils import set_seed, load_config, save_json, get_device
from src.data_setup import CIFAR10_CLASSES, load_cifar10_splits, get_or_create_q4_subset
from src.models import ConvAutoencoder
from src.train_eval import evaluate_autoencoder


def extract_features(cfg, data, device):
    """Extract fixed frozen 256-d latent representations from the Q3(b) DAE encoder."""
    bs = cfg["q3"]["batch_size"]
    ae = ConvAutoencoder(latent_dim=256).to(device)
    ae.load_state_dict(torch.load("checkpoints/q3b_dae_256.pt", map_location=device))

    z_tr = evaluate_autoencoder(ae, data["x_train_01"], data["x_train_01"], bs, device)["latents"]
    z_va = evaluate_autoencoder(ae, data["x_val_01"], data["x_val_01"], bs, device)["latents"]
    z_te = evaluate_autoencoder(ae, data["x_test_01"], data["x_test_01"], bs, device)["latents"]
    return z_tr, z_va, z_te


def run_q4a_and_b(cfg, data, z_tr_sub, y_tr_sub, z_va, y_va, z_te, y_te, sub_idx):
    print("\n=== Q4(a): SVM C and Kernel Sweep on Fixed Representation ===")
    c_grid = cfg["q4"]["C_grid"]
    kernels = cfg["q4"]["kernels"]
    poly_deg = cfg["q4"]["poly_degree"]
    seed = cfg["student"]["seed"]

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(z_tr_sub)
    x_va = scaler.transform(z_va)
    x_te = scaler.transform(z_te)

    sweep = []
    best_val_acc = -1.0
    best_cfg = None
    best_svm = None

    for kernel in kernels:
        for c_val in c_grid:
            t0 = time.time()
            kwargs = {"C": c_val, "kernel": kernel, "gamma": "scale", "cache_size": 1000, "random_state": seed}
            if kernel == "poly":
                kwargs["degree"] = poly_deg
            clf = SVC(**kwargs)
            clf.fit(x_tr, y_tr_sub)

            tr_acc = float(clf.score(x_tr, y_tr_sub))
            va_acc = float(clf.score(x_va, y_va))
            sv_frac = float(len(clf.support_) / len(x_tr))
            elapsed = time.time() - t0

            row = {
                "kernel": kernel,
                "C": float(c_val),
                "train_acc": tr_acc,
                "val_acc": va_acc,
                "sv_count": int(len(clf.support_)),
                "sv_fraction": sv_frac,
                "fit_seconds": round(elapsed, 2),
            }
            sweep.append(row)
            print(
                f"  kernel={kernel:6s} C={c_val:7.2f} | "
                f"Train {tr_acc:.4f} | Val {va_acc:.4f} | SV frac {sv_frac:.3f} ({elapsed:.1f}s)"
            )

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_cfg = {"kernel": kernel, "C": float(c_val)}
                best_svm = clf

    # Evaluate selected SVM once on official test set
    te_preds = best_svm.predict(x_te)
    te_acc = float((te_preds == y_te).mean())
    cm_test = confusion_matrix(y_te, te_preds, labels=list(range(10)))
    per_class_acc = {
        CIFAR10_CLASSES[c]: float((te_preds[y_te == c] == c).mean()) for c in range(10)
    }

    print(
        f"\nSelected SVM: kernel={best_cfg['kernel']}, C={best_cfg['C']} | "
        f"Val Acc={best_val_acc:.4f} | Test Acc={te_acc:.4f}"
    )

    # --- Q4(b): Difficult Class Pairs, Support Vectors, and Near-Boundary Test Images ---
    print("\n=== Q4(b): Difficult Class Pairs, Support Vectors & Boundary Analysis ===")
    # Rank unordered class pairs (i < j) by total mutual test confusion errors cm[i,j] + cm[j,i]
    pair_errors = []
    for i in range(10):
        for j in range(i + 1, 10):
            err_ij = int(cm_test[i, j] + cm_test[j, i])
            pair_errors.append((err_ij, i, j))
    pair_errors.sort(reverse=True)
    difficult_pairs = pair_errors[:4]

    # Map one-vs-one decision function columns: sklearn SVC orders pairs (i, j) with 0 <= i < j < K
    pair_to_col = {}
    col = 0
    for i in range(10):
        for j in range(i + 1, 10):
            pair_to_col[(i, j)] = col
            col += 1

    # Raw OVO decision scores (positive -> class i, negative -> class j)
    df_test = best_svm.decision_function(x_te)  # (N_test, 45)
    # Also train binary SVMs only or extract SV indices directly from multiclass SVC!
    # Wait: best_svm.support_ gives indices into x_tr; evaluate decision_function on SVs too.
    sv_indices_global = best_svm.support_
    sv_x = x_tr[sv_indices_global]
    sv_y = y_tr_sub[sv_indices_global]
    df_sv = best_svm.decision_function(sv_x)

    # Set SVC decision_function_shape='ovo' check (default is 'ovr', so pass explicit binary SVC or compute OVO!)
    # Note: SVC default decision_function_shape='ovr' returns shape (N, 10)!
    # Let's refit/clone with decision_function_shape='ovo' OR fit dedicated OVO binary check?
    # Better: fit the exact same best_svm with decision_function_shape='ovo' (the underlying dual solution is identical!).
    svm_ovo = SVC(
        C=best_cfg["C"],
        kernel=best_cfg["kernel"],
        gamma="scale",
        degree=poly_deg if best_cfg["kernel"] == "poly" else 3,
        cache_size=1000,
        decision_function_shape="ovo",
        random_state=seed,
    )
    svm_ovo.fit(x_tr, y_tr_sub)
    df_test_ovo = svm_ovo.decision_function(x_te)
    df_sv_ovo = svm_ovo.decision_function(x_tr[svm_ovo.support_])

    pair_details = []
    visual_payload = {}

    for err_count, ci, cj in difficult_pairs:
        cname_i, cname_j = CIFAR10_CLASSES[ci], CIFAR10_CLASSES[cj]
        c_col = pair_to_col[(ci, cj)]

        # Support vectors belonging to class ci or cj
        sv_mask_pair = (sv_y == ci) | (sv_y == cj)
        sv_pair_local = np.where(sv_mask_pair)[0]
        sv_pair_sub_idx = svm_ovo.support_[sv_pair_local]
        sv_pair_scores = df_sv_ovo[sv_pair_local, c_col]
        # Pick SVs closest to the ci-vs-cj boundary (|score| near 0 or 1 margin)
        sv_order = np.argsort(np.abs(sv_pair_scores))[:6]

        sv_records = []
        sv_imgs = []
        for k in sv_order:
            sub_pos = int(sv_pair_sub_idx[k])
            orig_train_pos = int(sub_idx[sub_pos])
            sv_records.append({
                "subset_index": sub_pos,
                "train_index": orig_train_pos,
                "true_label": CIFAR10_CLASSES[int(y_tr_sub[sub_pos])],
                "ovo_score": float(sv_pair_scores[k]),
            })
            sv_imgs.append(data["x_train_01"][orig_train_pos])

        # Test images belonging to class ci or cj closest to the ci-vs-cj OVO boundary
        test_mask_pair = (y_te == ci) | (y_te == cj)
        test_pair_idx = np.where(test_mask_pair)[0]
        test_pair_scores = df_test_ovo[test_pair_idx, c_col]
        near_order = np.argsort(np.abs(test_pair_scores))[:6]

        near_records = []
        near_imgs = []
        for k in near_order:
            t_idx = int(test_pair_idx[k])
            near_records.append({
                "test_index": t_idx,
                "true_label": CIFAR10_CLASSES[int(y_te[t_idx])],
                "pred_label": CIFAR10_CLASSES[int(te_preds[t_idx])],
                "ovo_score": float(test_pair_scores[k]),
                "correct": bool(te_preds[t_idx] == y_te[t_idx]),
            })
            near_imgs.append(data["x_test_01"][t_idx])

        # Compute pairwise Euclidean distance ratio between class centroids in scaled space
        mu_i = x_tr[y_tr_sub == ci].mean(axis=0)
        mu_j = x_tr[y_tr_sub == cj].mean(axis=0)
        centroid_dist = float(np.linalg.norm(mu_i - mu_j))
        # Mean L2 spread within classes
        spread_i = float(np.mean(np.linalg.norm(x_tr[y_tr_sub == ci] - mu_i, axis=1)))
        spread_j = float(np.mean(np.linalg.norm(x_tr[y_tr_sub == cj] - mu_j, axis=1)))

        pair_key = f"{cname_i}_vs_{cname_j}"
        print(
            f"  Pair {pair_key:20s}: mutual errors={err_count:3d} "
            f"(cm[{cname_i}->{cname_j}]={cm_test[ci, cj]}, cm[{cname_j}->{cname_i}]={cm_test[cj, ci]}) | "
            f"centroid_dist={centroid_dist:.2f}, avg_spread={(spread_i+spread_j)/2:.2f}"
        )

        pair_details.append({
            "pair": pair_key,
            "class_i": cname_i,
            "class_j": cname_j,
            "mutual_test_errors": err_count,
            "err_i_to_j": int(cm_test[ci, cj]),
            "err_j_to_i": int(cm_test[cj, ci]),
            "centroid_dist": centroid_dist,
            "spread_i": spread_i,
            "spread_j": spread_j,
            "support_vectors": sv_records,
            "near_boundary_test": near_records,
        })
        visual_payload[f"{pair_key}_sv_imgs"] = np.stack(sv_imgs, axis=0)
        visual_payload[f"{pair_key}_test_imgs"] = np.stack(near_imgs, axis=0)

    return (
        {
            "sweep": sweep,
            "selected": best_cfg,
            "selected_val_acc": best_val_acc,
            "test_acc": te_acc,
            "test_per_class_acc": per_class_acc,
            "test_confusion_matrix": cm_test.tolist(),
        },
        {
            "difficult_pairs": pair_details,
        },
        te_preds,
        visual_payload,
    )


def run_q4c(cfg, z_tr, y_tr, z_va, y_va, z_te, y_te, svm_test_preds):
    print("\n=== Q4(c): Unsupervised Clustering vs Supervised SVM ===")
    seed = cfg["student"]["seed"]
    k_grid = cfg["q4"]["cluster_k_grid"]

    # Fit scaler on full 45,000 train representations (no labels used!)
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(z_tr)
    x_va = scaler.transform(z_va)
    x_te = scaler.transform(z_te)

    # Subsample 10,000 train points (fixed seed) solely to compute Silhouette score cleanly
    rng = np.random.default_rng(seed)
    sil_idx = rng.choice(len(x_tr), size=10000, replace=False)

    model_selection = []
    best_score = -1.0
    best_method = None
    best_k = None

    for method in ["kmeans", "gmm_diag"]:
        for k in k_grid:
            if method == "kmeans":
                clus = KMeans(n_clusters=k, n_init=10, random_state=seed)
                clus.fit(x_tr)
                tr_labels = clus.predict(x_tr[sil_idx])
                va_labels = clus.predict(x_va)
            else:
                clus = GaussianMixture(
                    n_components=k, covariance_type="diag", n_init=3, max_iter=150, random_state=seed
                )
                clus.fit(x_tr)
                tr_labels = clus.predict(x_tr[sil_idx])
                va_labels = clus.predict(x_va)

            tr_sil = float(silhouette_score(x_tr[sil_idx], tr_labels))
            va_sil = float(silhouette_score(x_va, va_labels))
            va_db = float(davies_bouldin_score(x_va, va_labels))

            model_selection.append({
                "method": method,
                "k": k,
                "train_silhouette": tr_sil,
                "val_silhouette": va_sil,
                "val_davies_bouldin": va_db,
            })
            print(
                f"  {method:8s} k={k:2d} | Train Sil={tr_sil:.4f} | "
                f"Val Sil={va_sil:.4f} | Val DB={va_db:.4f}"
            )

            if va_sil > best_score:
                best_score = va_sil
                best_method = method
                best_k = k

    # Also fit k=10 KMeans as a direct class-matched reference alongside selected (best_method, best_k)
    def eval_clustering_on_test(method_name, k_val):
        if method_name == "kmeans":
            m = KMeans(n_clusters=k_val, n_init=20, random_state=seed)
            m.fit(x_tr)
        else:
            m = GaussianMixture(
                n_components=k_val, covariance_type="diag", n_init=5, max_iter=200, random_state=seed
            )
            m.fit(x_tr)
        # Assign test clusters WITHOUT test labels
        test_clusters = m.predict(x_te)
        # Use test labels ONLY after assignment for evaluation
        ari = float(adjusted_rand_score(y_te, test_clusters))
        nmi = float(normalized_mutual_info_score(y_te, test_clusters))

        contingency = np.zeros((k_val, 10), dtype=np.int64)
        for c_id, y_id in zip(test_clusters, y_te):
            contingency[int(c_id), int(y_id)] += 1

        # Majority-vote purity & per-class concentration across clusters
        cluster_purity = float(contingency.max(axis=1).sum() / len(y_te))
        class_dominant_cluster = {}
        for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
            col_counts = contingency[:, cls_idx]
            dom_c = int(col_counts.argmax())
            dom_frac = float(col_counts[dom_c] / col_counts.sum())
            # Entropy of how dispersed this class is across clusters
            p = col_counts / col_counts.sum()
            p_nz = p[p > 0]
            entropy = float(-np.sum(p_nz * np.log2(p_nz)))
            class_dominant_cluster[cls_name] = {
                "dominant_cluster": dom_c,
                "dominant_fraction": dom_frac,
                "cluster_entropy_bits": entropy,
                "svm_test_acc": float((svm_test_preds[y_te == cls_idx] == cls_idx).mean()),
            }

        # Pairwise cluster overlap: fraction of class i and class j sharing the same dominant cluster
        # or Bhattacharyya / cosine similarity of their cluster-distribution vectors
        class_dist_vecs = contingency.T.astype(np.float64)
        class_dist_vecs /= class_dist_vecs.sum(axis=1, keepdims=True)
        pair_disagreements = []
        cm_svm = confusion_matrix(y_te, svm_test_preds, labels=list(range(10)))
        for i in range(10):
            for j in range(i + 1, 10):
                # Cluster overlap (sum_k min(p_ik, p_jk))
                overlap = float(np.minimum(class_dist_vecs[i], class_dist_vecs[j]).sum())
                svm_pair_err = float((cm_svm[i, j] + cm_svm[j, i]) / 2000.0)
                pair_disagreements.append({
                    "pair": f"{CIFAR10_CLASSES[i]}_vs_{CIFAR10_CLASSES[j]}",
                    "cluster_distribution_overlap": overlap,
                    "svm_pairwise_error_rate": svm_pair_err,
                    "svm_pairwise_accuracy": 1.0 - svm_pair_err,
                })
        pair_disagreements.sort(key=lambda d: d["cluster_distribution_overlap"], reverse=True)

        return {
            "method": method_name,
            "k": k_val,
            "test_ari": ari,
            "test_nmi": nmi,
            "cluster_purity": cluster_purity,
            "contingency_table": contingency.tolist(),
            "per_class_summary": class_dominant_cluster,
            "top_cluster_overlaps": pair_disagreements[:10],
        }

    selected_test = eval_clustering_on_test(best_method, best_k)
    ref_k10_test = (
        selected_test
        if (best_method == "kmeans" and best_k == 10)
        else eval_clustering_on_test("kmeans", 10)
    )

    print(
        f"\nSelected Clustering ({best_method}, k={best_k}): "
        f"Test ARI={selected_test['test_ari']:.4f}, NMI={selected_test['test_nmi']:.4f}, "
        f"Purity={selected_test['cluster_purity']:.4f}"
    )
    if ref_k10_test is not selected_test:
        print(
            f"Reference KMeans (k=10): Test ARI={ref_k10_test['test_ari']:.4f}, "
            f"NMI={ref_k10_test['test_nmi']:.4f}, Purity={ref_k10_test['cluster_purity']:.4f}"
        )

    return {
        "model_selection": model_selection,
        "selected": {"method": best_method, "k": best_k},
        "selected_test_eval": selected_test,
        "reference_kmeans_10_eval": ref_k10_test,
    }


def main():
    cfg = load_config()
    seed = cfg["student"]["seed"]
    device = get_device()
    print(f"Using device: {device}")

    data = load_cifar10_splits(cfg)
    sub_idx = get_or_create_q4_subset(
        data["y_train"],
        cfg["data"]["q4_subset_file"],
        subset_size=cfg["q4"]["subset_size"],
        seed=seed,
    )

    z_tr, z_va, z_te = extract_features(cfg, data, device)
    z_tr_sub = z_tr[sub_idx]
    y_tr_sub = data["y_train"][sub_idx]

    q4a_res, q4b_res, svm_te_preds, visual_payload = run_q4a_and_b(
        cfg, data, z_tr_sub, y_tr_sub, z_va, data["y_val"], z_te, data["y_test"], sub_idx
    )
    q4c_res = run_q4c(cfg, z_tr, data["y_train"], z_va, data["y_val"], z_te, data["y_test"], svm_te_preds)

    np.savez("logs/q4_visuals.npz", **visual_payload)
    save_json({"q4a": q4a_res, "q4b": q4b_res, "q4c": q4c_res}, "logs/q4_results.json")
    print("\nSaved Q4 results to logs/q4_results.json and visuals to logs/q4_visuals.npz")


if __name__ == "__main__":
    main()
