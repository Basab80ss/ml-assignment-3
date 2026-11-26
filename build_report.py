import argparse
import os
import subprocess
import base64
from src.utils import load_config, load_json


def img_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "pending"


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def build_html(cfg: dict, q1: dict, q2: dict, q3: dict, q4: dict, commit_hash: str, repo_url: str) -> str:
    figs = {f"fig{i}": img_to_data_uri(p) for i, p in enumerate([
        "figures/fig1_q1a_mlp_comparison.png",
        "figures/fig2_q1b_memorisation.png",
        "figures/fig3_q2_cnn_vs_mlp.png",
        "figures/fig4_q2c_confidence_errors.png",
        "figures/fig5_q3ab_ae_denoising.png",
        "figures/fig6_q3bc_visuals_probes.png",
        "figures/fig7_q4a_svm_sweep.png",
        "figures/fig8_q4b_boundary_examples.png",
        "figures/fig9_q4c_clustering_vs_svm.png",
    ], start=1)}

    # Extract Q1 tables
    q1a_rows = ""
    for name in ["MLP-Plain", "MLP-ResBN", "MLP-Reg", "MLP-ShiftAug"]:
        r = q1["q1a"][name]
        q1a_rows += (
            f"<tr><td>{name}</td><td>{r['params']:,}</td>"
            f"<td>{fmt_pct(r['final_train_acc'])}</td>"
            f"<td>{fmt_pct(r['best_val_acc'])} (ep {r['best_epoch']})</td>"
            f"<td><b>{fmt_pct(r['test_acc'])}</b></td></tr>"
        )

    h40 = q1["q1b"]["noisy_40pct"]["history"]
    q1b_snap_rows = ""
    for ep in [1, 4, 6, 10, 15, 25, 35]:
        i = ep - 1
        q1b_snap_rows += (
            f"<tr><td>{ep}</td><td>{fmt_pct(h40['train_clean_acc'][i])}</td>"
            f"<td>{fmt_pct(h40['train_noisy_acc'][i])}</td>"
            f"<td>{fmt_pct(h40['val_acc'][i])}</td>"
            f"<td>{h40['val_loss'][i]:.3f}</td>"
            f"<td>{h40['weight_norm'][i]:.1f}</td></tr>"
        )

    # Extract Q2 table
    q2_rows = ""
    cond_names = [("clean", "Clean"), ("pixel_permuted", "Pixel-Permuted"), ("patch_8x8_permuted", "8x8 Patch-Permuted")]
    for ck, clbl in cond_names:
        for mk in ["FairCNN", "FairMLP"]:
            r = q2["conditions"][ck][mk]
            a = r["analysis"]
            q2_rows += (
                f"<tr><td>{clbl}</td><td>{mk}</td><td>{r['params']:,}</td>"
                f"<td>{fmt_pct(r['best_val_acc'])}</td><td><b>{fmt_pct(r['test_acc'])}</b></td>"
                f"<td>{a['mean_conf_correct']:.3f} / {a['mean_conf_wrong']:.3f}</td>"
                f"<td>{a['ece']:.3f}</td></tr>"
            )

    # Extract Q3 tables
    q3a_rows = ""
    for d in ["16", "64", "256"]:
        r = q3["q3a"]["models"][d]
        sel_mark = " *" if int(d) == q3["q3a"]["selected_dim"] else ""
        te_str = f"{q3['q3a']['selected_test_mse']:.5f} ({q3['q3a']['selected_test_psnr']:.2f})" if sel_mark else "&mdash;"
        q3a_rows += (
            f"<tr><td>AE-{d}{sel_mark}</td><td>{r['params']:,}</td>"
            f"<td>{r['val_mse']:.5f}</td><td>{r['val_psnr']:.2f} dB</td><td>{te_str}</td></tr>"
        )

    q3b_cls = q3["q3b"]["test_classification"]
    q3c_rows = ""
    for name in ["AE-16", "AE-64", "AE-256", "DAE-256"]:
        r = q3["q3c"]["clean_image_probes"][name]
        q3c_rows += (
            f"<tr><td>{name}</td><td>{r['test_clean_mse']:.5f}</td><td>{r['test_clean_psnr']:.2f} dB</td>"
            f"<td>{r['probe']['best_C']}</td><td>{fmt_pct(r['probe']['best_val_acc'])}</td>"
            f"<td><b>{fmt_pct(r['probe']['test_acc'])}</b></td></tr>"
        )
    dn_probe = q3["q3c"]["dae_noisy_image_probe"]
    q3c_rows += (
        f"<tr><td>DAE-256 (Noisy In)</td><td>{dn_probe['test_mse_vs_clean']:.5f}</td>"
        f"<td>{q3['q3b']['test_recon_dae_on_noisy']['psnr']:.2f} dB</td>"
        f"<td>{dn_probe['probe']['best_C']}</td><td>{fmt_pct(dn_probe['probe']['best_val_acc'])}</td>"
        f"<td><b>{fmt_pct(dn_probe['probe']['test_acc'])}</b></td></tr>"
    )

    # Extract Q4 SVM sweep table
    q4a_rows = ""
    sweep_by_kc = {(r["kernel"], r["C"]): r for r in q4["q4a"]["sweep"]}
    for c_val in cfg["q4"]["C_grid"]:
        cells = []
        for k in ["linear", "rbf", "poly"]:
            r = sweep_by_kc[(k, float(c_val))]
            is_sel = (k == q4["q4a"]["selected"]["kernel"] and float(c_val) == q4["q4a"]["selected"]["C"])
            val_txt = f"<b>{fmt_pct(r['val_acc'])}*</b>" if is_sel else fmt_pct(r["val_acc"])
            cells.append(f"<td>{fmt_pct(r['train_acc'])}</td><td>{val_txt}</td><td>{r['sv_fraction']:.3f}</td>")
        q4a_rows += f"<tr><td>{c_val:g}</td>{''.join(cells)}</tr>"

    # Extract Q4(c) contingency table for selected KMeans (k=5)
    sel_c = q4["q4c"]["selected_test_eval"]
    cont_rows = ""
    for ci, row in enumerate(sel_c["contingency_table"]):
        tds = "".join(f"<td>{v}</td>" for v in row)
        cont_rows += f"<tr><td><b>C{ci}</b></td>{tds}<td>{sum(row)}</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ML Assignment 3 — m26cse002</title>
<style>
  @page {{
    size: letter;
    margin: 0.55in 0.52in 0.58in 0.52in;
    @bottom-center {{
      content: "Page " counter(page);
      font-family: "Times New Roman", Times, serif;
      font-size: 8.5pt;
      color: #444;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Times New Roman", Times, serif;
    font-size: 9.2pt;
    line-height: 1.28;
    color: #111;
    margin: 0;
  }}
  .header {{
    text-align: center;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1.2px solid #222;
  }}
  .title {{
    font-size: 14.5pt;
    font-weight: bold;
    margin-bottom: 4px;
    line-height: 1.15;
  }}
  .authors {{
    font-size: 9.8pt;
    margin-bottom: 2px;
  }}
  .meta {{
    font-size: 8.4pt;
    color: #333;
    font-family: "Courier New", monospace;
  }}
  .two-col {{
    column-count: 2;
    column-gap: 0.22in;
    text-align: justify;
  }}
  h2 {{
    font-size: 10.6pt;
    font-weight: bold;
    margin: 8px 0 3px 0;
    border-bottom: 0.6px solid #bbb;
    padding-bottom: 1px;
    break-after: avoid;
  }}
  h3 {{
    font-size: 9.5pt;
    font-weight: bold;
    margin: 6px 0 2px 0;
    break-after: avoid;
  }}
  p {{ margin: 0 0 5px 0; }}
  ul {{ margin: 2px 0 5px 0; padding-left: 14px; }}
  li {{ margin-bottom: 2px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 5px 0 6px 0;
    font-size: 7.7pt;
    break-inside: avoid;
  }}
  th, td {{
    border-top: 0.5px solid #888;
    border-bottom: 0.5px solid #888;
    padding: 2.2px 3.2px;
    text-align: center;
  }}
  th {{
    background: #f2f4f7;
    font-weight: bold;
    border-top: 1.1px solid #222;
    border-bottom: 1.0px solid #222;
  }}
  td:first-child, th:first-child {{ text-align: left; }}
  .fig {{
    margin: 5px 0 6px 0;
    text-align: center;
    break-inside: avoid;
  }}
  .fig img {{
    width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
  }}
  .caption {{
    font-size: 7.8pt;
    color: #222;
    text-align: left;
    margin-top: 2px;
    line-height: 1.22;
  }}
  .span-all {{
    column-span: all;
    margin: 6px 0;
    break-inside: avoid;
  }}
  .box {{
    background: #f7f8fa;
    border-left: 2.5px solid #2c5282;
    padding: 4px 6px;
    margin: 4px 0 6px 0;
    font-size: 8.5pt;
    break-inside: avoid;
  }}
  code {{
    font-family: "Courier New", monospace;
    font-size: 8.0pt;
    background: #f3f3f3;
    padding: 0 2px;
  }}
  .refs {{
    font-size: 8.2pt;
    line-height: 1.25;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="title">Empirical Investigation of Inductive Bias, Memorisation, Autoencoding,<br>Kernel SVMs, and Clustering on CIFAR-10</div>
  <div class="authors"><b>Basab Bose</b> &nbsp;&bull;&nbsp; Roll No: <b>m26cse002</b> &nbsp;&bull;&nbsp; <code>m26cse002@iitj.ac.in</code></div>
  <div class="meta">GitHub: {repo_url} &nbsp;|&nbsp; Final Commit: <b>{commit_hash}</b> &nbsp;|&nbsp; Seed: 26002</div>
</div>

<div class="two-col">

<div class="box">
  <b>Abstract.</b> We present a controlled empirical study across four complementary investigations on CIFAR-10 using a fixed stratified 45,000/5,000 train/validation split (seed <code>26002</code>). First, parameter-matched MLP ablations (~1.74M parameters) isolate spatial coordinate overfitting from optimisation and regularisation bottlenecks, and a 40% label-noise trajectory demonstrates that clean generalisable structure peaks at Epoch 6 before noisy memorisation degrades test accuracy. Second, a parameter-matched CNN vs. MLP comparison (~214k parameters) across clean, pixel-permuted, and 8&times;8 patch-permuted inputs separates local texture extraction (+14.26% over MLP) from global spatial layout (+10.23%). Third, convolutional autoencoders (latent dimensions 16, 64, 256) and a &sigma;=0.25 denoising autoencoder reveal that pixel-MSE reconstruction quality and frozen linear-probe class separability diverge between clean AE-256 (23.67 dB, 47.33%) and DAE-256 (22.07 dB, 48.80%). Finally, kernel SVMs and unsupervised clustering on the frozen DAE-256 representation uncover sharp geometric vs. discriminative disagreements, such as 85.9% cluster overlap vs. 97.5% RBF-SVM separation between <i>automobile</i> and <i>horse</i>.
</div>

<h2>1. Common Setup and Reproducibility</h2>
<p>All experiments use CIFAR-10 (50,000 training and 10,000 official test 32&times;32 RGB images across 10 classes). Following the assignment protocol, we extract all numeric digits of roll number <code>m26cse002</code> in order to set <code>SEED = 26002</code> and construct one stratified split of 45,000 training images (4,500/class) and 5,000 validation images (500/class), saved in <code>splits/cifar10_split_26002.npz</code>. Channel-wise mean and standard deviation are computed strictly on the 45,000 training split and applied identically to validation and test data in Q1&ndash;Q2. All architecture choices, epoch selections, and regularisation parameters are chosen exclusively on validation metrics; the official 10,000-image test set is evaluated once per fixed model.</p>

<h2>2. Q1: Why MLPs Struggle &amp; Memorisation Dynamics</h2>

<h3>2.1 Q1(a): Separating Explanations Under a Fair Budget</h3>
<p><b>Hypothesis.</b> We test why standard MLPs underperform on CIFAR-10 by isolating <i>spatial coordinate sensitivity / pixel-template memorisation</i> from two plausible alternatives: (1) <i>optimisation difficulty</i> (poor gradient flow without normalisation or skip connections) and (2) <i>generic capacity overfitting</i> (excess unregularised parameters). To keep the comparison fair, all four models share an identical ~1.74M parameter budget, AdamW (<code>lr=1e-3</code>), cosine annealing over 25 epochs, batch size 128, and the same 45k/5k split:</p>
<ul>
  <li><b>MLP-Plain</b> (1,738,890 params): <code>3072-512-256-128-10</code>, ReLU, weight decay <code>1e-4</code>.</li>
  <li><b>MLP-ResBN</b> (1,747,250 params, within 0.48%): <code>3072-440</code> stem + two BatchNorm residual blocks (<code>440&rarr;440</code>), testing optimisation/gradient flow.</li>
  <li><b>MLP-Reg</b> (1,738,890 params): identical to MLP-Plain with Dropout <code>0.35</code> and weight decay <code>1e-2</code>, testing generic regularisation.</li>
  <li><b>MLP-ShiftAug</b> (1,738,890 params): identical to MLP-Plain trained with random 4-px padded crops and horizontal flips, directly testing sensitivity to exact pixel coordinates.</li>
</ul>

<table>
  <tr><th>Model</th><th>Params</th><th>Final Train</th><th>Best Val (Ep)</th><th>Test Acc</th></tr>
  {q1a_rows}
</table>

<div class="fig">
  <img src="{figs['fig1']}" alt="Q1a MLP Comparison">
  <div class="caption"><b>Figure 1:</b> Q1(a) ~1.74M-parameter MLP comparison. Fixing optimisation (<code>MLP-ResBN</code>) achieves 99.41% train accuracy yet gains only +1.60% on test; heavy regularisation (<code>MLP-Reg</code>) shrinks the train&ndash;val gap to 5.44% yet stalls at 55.68% test.</div>
</div>

<p><b>Interpretation.</b> Table 1 and Fig. 1 cleanly separate the hypotheses. <code>MLP-Plain</code> reaches 99.56% training accuracy, and improving optimisation (<code>MLP-ResBN</code>) accelerates convergence (55.30% vs. 52.76% val at epoch 5) while still saturating at <b>55.88%</b> test accuracy&mdash;ruling out underfitting/optimisation failure. Suppressing memorisation via Dropout+L2 (<code>MLP-Reg</code>) slashes training accuracy from 99.56% to 62.18% but yields only <b>55.68%</b> test accuracy, proving that generic capacity control alone cannot discover spatial features. Forcing modest shift tolerance (<code>MLP-ShiftAug</code>) provides the largest gain (<b>57.09%</b> test), yet still hits a hard representational ceiling below 58% because dense input weights cannot share local detectors across spatial locations.</p>

<h3>2.2 Q1(b): Structure Learning Before Memorisation</h3>
<p>To construct a setting capable of fitting non-generalisable targets, we use <code>MLPMem</code> (<code>3072-768-384-192-10</code>, 2,731,210 parameters, zero dropout, zero weight decay, constant Adam <code>lr=1e-3</code>, 35 epochs) under three training label conditions while keeping the 5,000 validation set 100% clean: <b>0% noise</b>, <b>40% wrong-label noise</b> (18,000 training labels replaced uniformly by wrong classes), and <b>100% random labels</b>.</p>

<table>
  <tr><th>Epoch</th><th>Train Clean (60%)</th><th>Train Noisy (40%)</th><th>Clean Val Acc</th><th>Val Loss</th><th>Weight L2</th></tr>
  {q1b_snap_rows}
</table>

<div class="fig">
  <img src="{figs['fig2']}" alt="Q1b Memorisation Trajectory">
  <div class="caption"><b>Figure 2:</b> Q1(b) training trajectory under 40% label noise (left) and noise sweep (right). Useful structure peaks at Epoch 6 (46.90% val, 10.24% noisy-subset acc) before wrong-label memorisation climbs to 70.64% and degrades validation accuracy to 36.26%.</div>
</div>

<p><b>Evidence from the trajectory.</b> Under 40% noise, <code>MLPMem</code> clearly learns generalisable structure <i>before</i> memorising corrupted labels (Fig. 2a, Table 2). During Epochs 1&ndash;6, clean-subset training accuracy rises from 39.80% to 55.99% and clean validation accuracy peaks at <b>46.90%</b> (test <b>45.85%</b>), while noisy-subset accuracy stays near chance (<b>10.24%</b>, noisy loss 2.50). Only after Epoch 6 does memorisation accelerate: noisy-subset accuracy surges from 10.24% (Ep 6) to 48.40% (Ep 20) and <b>70.64%</b> (Ep 35), driving weight L2 norm from 68.9 to 177.9 and steadily eroding validation accuracy down to <b>36.26%</b> (final test <b>35.19%</b>). Meanwhile, under 100% random labels, training accuracy climbs to 21.55% (val 10.06%), confirming capacity without generalisable signal.</p>

<h2>3. Q2: Spatial Inductive Bias in CNNs vs. MLPs</h2>

<h3>3.1 Q2(a)&ndash;(b): Fair Comparison and Spatial Disturbance</h3>
<p><b>Fairness criterion (defined a priori).</b> We match trainable parameter count within 1%, normalisation, activation, and training budget: <b>FairCNN</b> (4 conv layers <code>3&rarr;24&rarr;48&rarr;96&rarr;128</code> with 3&times;3 kernels, BatchNorm, ReLU, MaxPool/AdaptiveAvgPool, <code>512&rarr;96&rarr;10</code> head, <b>214,378</b> params) vs. <b>FairMLP</b> (<code>3072-66-66-66-10</code> with BatchNorm and ReLU, <b>212,728</b> params, within 0.77%). Both train for 20 epochs with AdamW (<code>lr=1.5e-3</code>, <code>wd=1e-4</code>, batch 256, cosine LR, no augmentation). On clean CIFAR-10, <code>FairCNN</code> achieves <b>77.53%</b> test accuracy vs. <b>53.27%</b> for <code>FairMLP</code> (+24.26%). We propose that this gap stems from 2D local receptive fields and weight sharing exploiting local spatial adjacency.</p>
<p><b>Pre-registered prediction for Q2(b).</b> Applying one fixed random permutation of the 1,024 spatial <code>(y,x)</code> coordinates (seed <code>26002</code>) identically to train, val, and test preserves every RGB pixel value while destroying 2D adjacency. Because <code>FairMLP</code> is permutation-invariant across input coordinates, we expect its accuracy to stay ~53%; because 3&times;3 convolutions depend on local adjacency, <code>FairCNN</code> should collapse to ~50%. Observed test results confirm this: <code>FairMLP</code> stays at <b>52.06%</b> (&minus;1.21%), while <code>FairCNN</code> collapses by <b>&minus;27.24%</b> to <b>50.29%</b> (worse than <code>FairMLP</code>).</p>

<h3>3.2 Q2(c): Local Textures vs. Global Layout &amp; Confidence</h3>
<p>Pixel permutation destroys both <i>local 3&times;3 edge/texture structure</i> and <i>global object geometry</i>. To distinguish which drives <code>FairCNN</code>'s clean advantage, we apply a fixed permutation of the 16 non-overlapping <b>8&times;8 patches</b> (seed <code>26003</code>) identically to train, val, and test. Every 8&times;8 local neighbourhood remains intact, while global object layout is scrambled.</p>

<table>
  <tr><th>Condition</th><th>Model</th><th>Params</th><th>Best Val</th><th>Test Acc</th><th>Conf (Corr/Wrong)</th><th>ECE</th></tr>
  {q2_rows}
</table>

<div class="fig">
  <img src="{figs['fig3']}" alt="Q2 CNN vs MLP">
  <div class="caption"><b>Figure 3:</b> Q2 matched-budget test accuracy across clean, pixel-permuted, and 8&times;8 patch-permuted CIFAR-10 (left), and <code>FairCNN</code> per-class accuracy comparing Clean vs. 8&times;8 Patch-Permuted inputs (right).</div>
</div>

<div class="fig">
  <img src="{figs['fig4']}" alt="Q2 Confidence and Calibration">
  <div class="caption"><b>Figure 4:</b> Q2(c) test calibration reliability curves (left) and mean softmax confidence on correct vs. incorrect predictions (right). Patch-permuted <code>FairCNN</code> retains 92.1% / 77.7% confidence, inflating ECE to 0.201.</div>
</div>

<p><b>Error, confidence, and linking Q2(a)&ndash;(c).</b> On 8&times;8 patch-permuted images, <code>FairMLP</code> remains invariant at <b>53.04%</b>, while <code>FairCNN</code> achieves <b>67.30%</b>&mdash;retaining <b>+14.26%</b> over <code>FairMLP</code> yet losing <b>&minus;10.23%</b> vs. Clean <code>FairCNN</code>. This quantifies both mechanisms: ~58% of the CNN's clean gain comes from local 8&times;8 texture/edge extraction, and ~42% from global cross-patch geometry. Crucially, error and confidence analysis (Fig. 3b, Fig. 4) reveals how these errors occur: classes defined by global silhouette/pose suffer most (<i>deer</i> &minus;16.2%, <i>truck</i> &minus;14.0%, <i>cat</i> &minus;10.9%, with <i>cat&rarr;dog</i> rising from 146 to 204 errors and <i>truck&rarr;auto</i> reaching 111), whereas texture/background classes resist scrambling (<i>airplane</i> &minus;3.4%). Moreover, patch-permuted <code>FairCNN</code> remains highly overconfident on errors (mean wrong confidence <b>0.777</b>, 52.57% of errors &ge;0.80 confidence, ECE jumping from <b>0.136</b> to <b>0.201</b>), whereas pixel-permuted <code>FairCNN</code> drops to 0.497 wrong confidence (ECE 0.081). Together, Q2(a)&ndash;(c) show that 2D convolutional locality unlocks sharp feature confidence (Q2a&rarr;b), much of which relies on local bag-of-patches cues even when global shape is broken (Q2c).</p>

<h2>4. Q3: Autoencoders, Denoising &amp; Class Information</h2>

<h3>4.1 Q3(a): Bottleneck Capacity Sweep</h3>
<p>We train three symmetric convolutional autoencoders (<code>3&rarr;32&rarr;64&rarr;128</code> stride-2 encoder, linear bottleneck <code>z &isin; &#123;16, 64, 256&#125;</code>, mirrored transposed-conv decoder with Sigmoid output on <code>[0,1]</code> images) for 15 epochs (AdamW <code>lr=1.5e-3</code>, <code>wd=1e-4</code>, batch 256, MSE loss). If reconstruction error is the sole criterion, we select <b>AE-256</b> via lowest validation MSE (<b>0.00432</b>, PSNR <b>23.64 dB</b>), yielding official test MSE <b>0.00430</b> (<b>23.67 dB</b>).</p>

<table>
  <tr><th>Model</th><th>Params</th><th>Val MSE</th><th>Val PSNR</th><th>Test MSE (PSNR)</th></tr>
  {q3a_rows}
</table>

<h3>4.2 Q3(b): Denoising Recovery for Fixed CNN</h3>
<p>We corrupt images with additive Gaussian noise (<code>&sigma;=0.25</code> on <code>[0,1]</code>, clipped to <code>[0,1]</code>; deterministic test noise seed <code>26403</code>) and train <b>DAE-256</b> (identical architecture to AE-256, fresh weights, noisy inputs &rarr; clean targets, no class labels). We evaluate the <b>fixed Q2(a) FairCNN</b> (checkpoint <code>q2a_fair_cnn.pt</code>, standardized via training mean/std) on clean, noisy, Clean-AE reconstructed, and DAE reconstructed test images:</p>
<ul>
  <li><b>Clean test images:</b> Acc = <b>{fmt_pct(q3b_cls['clean_images']['acc'])}</b> (Mean Conf {q3b_cls['clean_images']['mean_conf']:.3f}).</li>
  <li><b>Corrupted (&sigma;=0.25):</b> Acc collapses to <b>{fmt_pct(q3b_cls['corrupted_images']['acc'])}</b> (overconfident {q3b_cls['corrupted_images']['mean_conf']:.3f} due to high-frequency noise activating conv filters).</li>
  <li><b>Clean AE-256 recon:</b> Test MSE vs. clean = {q3['q3b']['test_recon_clean_ae_on_noisy']['mse']:.5f} ({q3['q3b']['test_recon_clean_ae_on_noisy']['psnr']:.2f} dB); FairCNN Acc = <b>{fmt_pct(q3b_cls['clean_ae_recons']['acc'])}</b>.</li>
  <li><b>Denoising DAE-256 recon:</b> Test MSE vs. clean = {q3['q3b']['test_recon_dae_on_noisy']['mse']:.5f} ({q3['q3b']['test_recon_dae_on_noisy']['psnr']:.2f} dB); FairCNN Acc = <b>{fmt_pct(q3b_cls['dae_recons']['acc'])}</b>.</li>
</ul>

<div class="fig">
  <img src="{figs['fig5']}" alt="Q3a and Q3b">
  <div class="caption"><b>Figure 5:</b> Q3(a) validation MSE across bottleneck sizes (left) and Q3(b) fixed Q2(a) <code>FairCNN</code> test accuracy on clean, &sigma;=0.25 noisy, Clean-AE, and Denoising-AE reconstructions (right).</div>
</div>

<p><b>Does denoising recover what the classifier needs?</b> Partially, but <i>not fully</i>. DAE-256 lowers noisy reconstruction MSE from 0.00844 to 0.00676 (+0.96 dB) and lifts fixed-CNN accuracy by <b>+23.35%</b> over raw corrupted images (and +1.23% over Clean-AE). However, 35.68% remains far below the 77.53% clean ceiling because MSE-optimal decoding blurs sharp edges and fine textures (Fig. 6a) that the clean-trained CNN relies on.</p>

<h3>4.3 Q3(c): Reconstruction vs. Class Information</h3>
<p>Freezing all four encoders (without fine-tuning), we extract latent codes and fit standardized <code>LogisticRegression</code> linear probes (tuning <code>C &isin; &#123;0.01..100&#125;</code> on validation only).</p>

<table>
  <tr><th>Encoder</th><th>Test MSE</th><th>Test PSNR</th><th>Best C</th><th>Val Acc</th><th>Test Probe</th></tr>
  {q3c_rows}
</table>

<div class="fig">
  <img src="{figs['fig6']}" alt="Q3b Visuals and Q3c Probes">
  <div class="caption"><b>Figure 6:</b> Q3(b) sample test reconstructions showing noise removal alongside MSE smoothing (left), and Q3(c) clean-test PSNR vs. frozen linear-probe accuracy (right).</div>
</div>

<p><b>Do reconstruction quality and class information agree?</b> Across capacity (<code>AE-16 &rarr; AE-64 &rarr; AE-256</code>), both metrics improve together (17.81&rarr;23.67 dB; 33.14%&rarr;47.33%). However, comparing <code>AE-256</code> and <code>DAE-256</code> shows they <b>do not strictly agree</b>: <code>DAE-256</code> has <i>worse</i> clean pixel reconstruction (22.07 dB vs. 23.67 dB) because denoising discards high-frequency pixel variance, yet achieves <i>higher</i> linear class separability (<b>48.80%</b> vs. 47.33%, and retains <b>46.96%</b> even on &sigma;=0.25 noisy inputs!). This directly confirms that pixel-MSE fidelity rewards low-level appearance variance, whereas denoising encourages semantic invariance.</p>

<h2>5. Q4 (Bonus): SVMs &amp; Clustering on DAE-256</h2>

<h3>5.1 Q4(a): Fixed Representation, C &amp; Kernel Sweep</h3>
<p>We keep the frozen unsupervised 256-d clean-image representation from Q3(b)'s <b>DAE-256</b> unchanged. Following the Q4 note, we draw one fixed stratified training subset of <b>5,000 samples</b> (500/class, seed <code>26002</code>, saved in <code>splits/q4_svm_subset_26002.npz</code>) prior to any model selection, standardize features on this subset, and sweep 6 values of <code>C</code> spanning 3.3 orders of magnitude (<code>0.005</code> to <code>10.0</code>) across <code>linear</code>, <code>rbf</code>, and <code>poly</code> (degree 3) kernels on the full 5,000 validation set.</p>

<table>
  <tr><th rowspan="2">C</th><th colspan="3">Linear</th><th colspan="3">RBF</th><th colspan="3">Poly (d=3)</th></tr>
  <tr><th>Tr</th><th>Val</th><th>SV</th><th>Tr</th><th>Val</th><th>SV</th><th>Tr</th><th>Val</th><th>SV</th></tr>
  {q4a_rows}
</table>

<div class="fig">
  <img src="{figs['fig7']}" alt="Q4a SVM Sweep">
  <div class="caption"><b>Figure 7:</b> Q4(a) SVM train/validation accuracy (left) and support-vector fraction (right) vs. C. RBF peaks at C=2.0 (48.44% val) before overfitting at C=10.0 (97.54% train, 47.44% val).</div>
</div>

<p><b>Underfitting, overfitting, and test result.</b> At <code>C=0.005&ndash;0.02</code>, heavy margin slack causes severe underfitting: RBF and Poly stay near 29% and 12&ndash;13% val accuracy with 99.8&ndash;100% of training points acting as support vectors. Increasing <code>C</code> shrinks SV fractions (Linear: 93.6%&rarr;83.9%; RBF: 100%&rarr;96.4%) and raises accuracy until Linear peaks at <code>C=0.5</code> (42.10% val) and RBF peaks at <b><code>C=2.0</code> (48.44% val)</b>. At <code>C=10.0</code>, both Linear (77.02% train / 40.38% val) and RBF (97.54% train / 47.44% val) overfit noisy margin points. Evaluating the validation-selected <b>RBF (C=2.0)</b> SVM once on the official 10,000-image test set yields <b>{fmt_pct(q4['q4a']['test_acc'])}</b> accuracy (best classes: <i>automobile</i> 59.9%, <i>ship</i> 59.9%, <i>airplane</i> 56.2%; hardest: <i>bird</i> 31.2%, <i>cat</i> 32.2%).</p>

<h3>5.2 Q4(b): Difficult Class Pairs &amp; Boundary Geometry</h3>
<p>From the test confusion matrix, the top difficult pairs are <b>cat vs. dog</b> (372 mutual errors: 193+179), <b>bird vs. deer</b> (361: 197+164), <b>automobile vs. truck</b> (345: 159+186), and <b>airplane vs. ship</b> (267: 148+119). Visually inspecting training support vectors and near-boundary test images with their one-vs-one decision scores (Fig. 8) explains these errors geometrically: in the standardized 256-d DAE space, inter-centroid Euclidean distances for <i>cat&ndash;dog</i> (<b>1.95</b>) and <i>bird&ndash;deer</i> (<b>1.96</b>) are 6&ndash;8&times; smaller than within-class radial spreads (<b>15.63</b> and <b>12.15</b>; <i>auto&ndash;truck</i>: dist 3.96 vs. spread 17.10). Consequently, near-boundary support vectors and test images lie in densely overlapping regions (|OVO score| &le; 0.005) where pose, background grass/sky, or vehicle boxiness dominate the unsupervised bottleneck.</p>

<div class="fig">
  <img src="{figs['fig8']}" alt="Q4b Boundary Images">
  <div class="caption"><b>Figure 8:</b> Q4(b) training support vectors (cols 1&ndash;3) and near-boundary test images (cols 4&ndash;6) for the three hardest pairs, with OVO decision scores (green=correct, red=misclassified).</div>
</div>

<h3>5.3 Q4(c): Unsupervised Clustering vs. Supervised SVM</h3>
<p>Clustering the exact same standardized DAE-256 representations without labels, we evaluate <code>KMeans</code> and diagonal <code>GMM</code> across <code>k &isin; &#123;5, 8, 10, 12, 15, 20&#125;</code> on training and validation sets using unsupervised Silhouette and Davies-Bouldin scores. Silhouette clearly prefers <b>KMeans with k=5</b> (Val Sil <b>0.0321</b>, Train Sil <b>0.0309</b> vs. negative Silhouette for k&ge;10 and all GMMs). Fitting on the 45,000 training representations and assigning test clusters without labels yields: <b>Test ARI = 0.0271</b>, <b>Test NMI = 0.0602</b>, <b>Purity = 0.1697</b> (and for a 10-cluster KMeans reference: <b>ARI = 0.0478</b>, <b>NMI = 0.0920</b>, <b>Purity = 0.2377</b>).</p>

<table>
  <tr><th>Clus</th><th>Air</th><th>Auto</th><th>Bird</th><th>Cat</th><th>Deer</th><th>Dog</th><th>Frog</th><th>Hors</th><th>Ship</th><th>Trk</th><th>Tot</th></tr>
  {cont_rows}
</table>

<div class="fig">
  <img src="{figs['fig9']}" alt="Q4c Clustering vs SVM">
  <div class="caption"><b>Figure 9:</b> Q4(c) KMeans (k=5) test contingency heatmap (left) and per-class RBF-SVM test accuracy vs. maximum cluster concentration (right).</div>
</div>

<p><b>Classification vs. clustering disagreement.</b> While <i>cat vs. dog</i> and <i>bird vs. deer</i> are difficult under both paradigms (top-2 cluster overlaps 0.897 and 0.918; top-2 SVM errors), supervised SVM and unsupervised clustering sharply disagree on classes separated by fine shape rather than global colour/background statistics. Specifically, in KMeans (k=5), <b><i>automobile</i> and <i>horse</i> share 85.9% cluster distribution overlap</b> (both peaking in Cluster C1: 38.3% of automobiles and 36.8% of horses), and in k=10 KMeans <i>cat</i> and <i>horse</i> share 77.4% overlap; yet the supervised RBF-SVM separates <i>automobile vs. horse</i> with <b>97.5% pairwise accuracy</b> (and <i>cat vs. horse</i> with 93.3%). Likewise, <i>frog</i> achieves 55.2% SVM accuracy (4th highest) despite 46.6% of frogs collapsing into C1 alongside 46.7% of <i>deer</i> and 41.3% of <i>bird</i>. Thus, isotropic Euclidean clustering in DAE-256 space aligns with global scene/luminance modes, whereas nonlinear RBF margins recover class-discriminative directions.</p>

<h2>References</h2>
<div class="refs">
  <p>[1] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Tech. Rep., Univ. of Toronto, 2009.<br>
  [2] C. Zhang et al., "Understanding deep learning requires rethinking generalization," <i>ICLR</i>, 2017.<br>
  [3] D. Arpit et al., "A closer look at memorization in deep networks," <i>ICML</i>, 2017.<br>
  [4] P. Vincent et al., "Extracting and composing robust features with denoising autoencoders," <i>ICML</i>, 2008.<br>
  [5] C. Cortes and V. Vapnik, "Support-vector networks," <i>Machine Learning</i>, 20(3):273&ndash;297, 1995.</p>
</div>

</div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default=None)
    parser.add_argument("--repo", default="https://github.com/Basab80se/ml-assignment-3-cifar10")
    args = parser.parse_args()

    cfg = load_config()
    q1 = load_json("logs/q1_results.json")
    q2 = load_json("logs/q2_results.json")
    q3 = load_json("logs/q3_results.json")
    q4 = load_json("logs/q4_results.json")

    commit_hash = args.commit or get_git_commit()
    html = build_html(cfg, q1, q2, q3, q4, commit_hash=commit_hash, repo_url=args.repo)

    os.makedirs("report", exist_ok=True)
    html_path = os.path.abspath("report/report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    pdf_out = os.path.abspath("report/ML_Assignment3_m26cse002.pdf")
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    subprocess.run([
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_out}",
        f"file:///{html_path.replace(os.sep, '/')}",
    ], check=True)
    print(f"Generated PDF: {pdf_out} (commit={commit_hash})")


if __name__ == "__main__":
    main()
