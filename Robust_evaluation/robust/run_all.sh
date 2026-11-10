#!/usr/bin/env bash
# Usage: FLOWS=/kaggle/working/flows_by_class MFR=/kaggle/working/mfr_dataset bash robust/run_all.sh
set -euo pipefail
: "${FLOWS:?}"; : "${MFR:?}"
CK="out/ft_seed0/checkpoint-best.pth out/ft_seed1/checkpoint-best.pth out/ft_seed2/checkpoint-best.pth"
python -m pytest robust/tests -q
python -m robust.eval_attacks --flows_root $FLOWS --mfr_root $MFR --ckpts $CK --goal evasion --out res/evasion
python -m robust.eval_attacks --flows_root $FLOWS --ckpts $CK --goal impersonation --out res/impersonation --no_pgd
python -m robust.shortcut_analysis --flows_root $FLOWS --ckpt out/ft_seed0/checkpoint-best.pth --out res/shortcut
python -m robust.window_groups --flows_root $FLOWS --ckpt out/ft_seed0/checkpoint-best.pth --out res/windows
python -m robust.rq1_calibration --flows_root $FLOWS --pretrained pre-trained_TACO.pth --ckpt out/ft_seed0/checkpoint-best.pth --out res/rq1
python -m robust.foundation_transfer --flows_root $FLOWS --pretrained pre-trained_TACO.pth --ckpts $CK --out res/foundation
python -m robust.adv_finetune --flows_root $FLOWS --init_ckpt out/ft_seed0/checkpoint-best.pth --holdout keepalive --out res/defence
