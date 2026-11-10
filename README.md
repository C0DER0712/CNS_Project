# TaCo Robustness Study

This repository contains a Kaggle notebook, a local attack and evaluation package, and generated results for studying the robustness of TaCo, a pre-trained network traffic classification model, on the ISCXVPN dataset.

The main entry point is [`CNS_Project.ipynb`](CNS_Project.ipynb). It prepares packet capture data, builds per-flow and MFR image datasets, fine-tunes TaCo classifiers, evaluates evasion and impersonation attacks, and runs analyses of representation quality and possible identifier shortcuts. The notebook compares pre-trained and from-scratch models, includes capture-disjoint and identifier-masked experiments, and bundles outputs after each stage.

The project combines two codebases with distinct roles: TaCo supplies the model architecture, training and upstream code; [`Robust_evaluation/`](Robust_evaluation/) supplies this project's packet-level attack implementations, threat model, evaluation pipeline, analysis tools, and tests. The evaluation code is local to this repository and is not part of the TaCo GitHub repository.

## Data and upstream code

- **Dataset:** [ISCXVPN on Kaggle](https://www.kaggle.com/datasets/cs23b2013/iscxvpn)
- **TaCo model code:** [NSSL-SJTU/TaCo on GitHub](https://github.com/NSSL-SJTU/TaCo)
- **Attack and evaluation code:** [`Robust_evaluation/`](Robust_evaluation/) in this repository

The notebook needs the ISCXVPN packet captures (`RAW_PCAP_DIR` path variable in notebook), a TaCo checkout (`TACO_SRC` path variable in notebook) containing `fine-tune/`, `data/`, and `pre-trained_TACO.pth`, and this repository's `Robust_evaluation/robust/` package (`ROBUST_SRC` path variable in notebook). In Kaggle, attach the dataset and TaCo checkout as notebook inputs, and upload `Robust_evaluation/` as an input (or zip its contents) so the notebook can locate files such as `pcap_ops.py` and `linear_probe.py`. The setup cell searches under `/kaggle/input`; set `RAW_PCAP_DIR`, `TACO_SRC`, and `ROBUST_SRC` there to override automatic discovery.

## Running the notebook

Run `CNS_Project.ipynb` in a GPU-enabled Kaggle environment. The default `FAST = False` setting runs the full study and is estimated by the notebook to take about 8–9 hours. Set `FAST = True` for a shorter smoke run (about 1.5 hours); smoke-run outputs are limited and are not intended as final study results.

The notebook installs its Python dependencies during setup, prepares flow and image datasets, trains/evaluates models, and writes the results bundle to `/kaggle/working/taco_robust_results.zip`. Results are also written beneath `Results/` in this repository; these include merged attack summaries, CSV tables, diagnostic JSON, logs, and plots.

## Experiment overview

- Evasion attacks (random, greedy, gradient, and a PGD reference) across modification budgets.
- Impersonation and matched pre-trained versus scratch comparisons.
- Capture-disjoint evaluation and retraining with network identifiers masked.
- Linear probes, cosine-similarity calibration, foundation transfer, shortcut checks, and window-group analyses.
- Adversarial fine-tuning followed by adaptive attack evaluation.

## Repository contents

- `CNS_Project.ipynb` — end-to-end experiment notebook and configuration.
- `Robust_evaluation/robust/` — packet-level perturbation operators, attack implementations, model evaluation, analyses, and tests.
- `Robust_evaluation/THREAT_MODEL.md` — attacker capabilities, goals, budgets, and validity assumptions.
- `Results/` — generated figures, logs, and machine-readable summaries from the study.

The dataset and TaCo model implementation remain upstream resources; this repository does not include the packet captures, TaCo source checkout, or pre-trained checkpoint.
