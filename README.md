# TaCo Robustness Study

This repository contains a Kaggle notebook and its generated results for evaluating the robustness of TaCo, a pre-trained network traffic classification model, on the ISCXVPN dataset.

The main entry point is [`CNS_Project.ipynb`](CNS_Project.ipynb). It prepares packet capture data, builds per-flow and MFR image datasets, fine-tunes TaCo classifiers, evaluates evasion and impersonation attacks, and runs analyses of representation quality and possible identifier shortcuts. The notebook compares pre-trained and from-scratch models, includes capture-disjoint and identifier-masked experiments, and bundles outputs after each stage.

## Data and upstream code

- **Dataset:** [ISCXVPN on Kaggle](https://www.kaggle.com/datasets/cs23b2013/iscxvpn)
- **TaCo source code:** [NSSL-SJTU/TaCo on GitHub](https://github.com/NSSL-SJTU/TaCo)

The notebook expects the ISCXVPN packet captures, a TaCo checkout containing `fine-tune/`, `attack/`, `data/`, and `pre-trained_TACO.pth`, plus the `taco_robust` code used by the experiments. In Kaggle, attach these as notebook inputs. The setup cell attempts to discover the inputs automatically; set `RAW_PCAP_DIR`, `TACO_SRC`, and `ROBUST_SRC` there to override discovery.

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
- `Results/` — generated figures, logs, and machine-readable summaries from the study.

The dataset and TaCo implementation are upstream resources; this repository links to them and does not include their source data or code.
