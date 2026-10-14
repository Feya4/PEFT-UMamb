#!/usr/bin/env bash
# run_all.sh — every experiment needed for the revision, 4 seeds each.
# Edit the data paths, then:   bash run_all.sh            (all)
#                              bash run_all.sh main       (Tables 1–4 only)
set -e
cd "$(dirname "$0")"
PY=/workdir1.8t/anaconda3/envs/swin_umamba/bin/python
SEEDS="0 1 2 3"

AMOS=/workdir1.8t/fei27/CGT/peft_umamba/peft_umamba_Amos/data/Dataset702_AbdomenMR
KVASIR=/workdir1.8t/fei27/CGT/peft_umamba/peft_umamba1/data/kvasir
ENDOVIS=/path/to/Dataset704_Endovis17          # <-- edit
CELL=/path/to/Dataset703_NeurIPSCell           # <-- edit

run() { $PY train.py "$@"; }
WHAT=${1:-all}

# ── Tables 1–4: PEFT-UMamba on all four datasets ──────────────────────────────
if [[ $WHAT == all || $WHAT == main ]]; then
  for s in $SEEDS; do
    run --dataset dataset702 --data_root $AMOS    --seed $s
    run --dataset kvasir     --data_root $KVASIR  --seed $s
    run --dataset dataset704 --data_root $ENDOVIS --seed $s
    run --dataset dataset703 --data_root $CELL    --seed $s
  done
fi

# ── Table 6: adaptation strategies (AMOS22 + Kvasir-SEG) ──────────────────────
if [[ $WHAT == all || $WHAT == table6 ]]; then
  for s in $SEEDS; do
    for m in frozen lora full_ft mona lorand e3va; do
      run --dataset dataset702 --data_root $AMOS   --method $m --seed $s
      run --dataset kvasir     --data_root $KVASIR --method $m --seed $s
    done
    run --dataset dataset702 --data_root $AMOS   --method lora --lora_rank 8 --exp_name dataset702_lora_r8_s$s --seed $s
    run --dataset kvasir     --data_root $KVASIR --method lora --lora_rank 8 --exp_name kvasir_lora_r8_s$s     --seed $s
  done
fi

# ── Table 5: component ablations ──────────────────────────────────────────────
if [[ $WHAT == all || $WHAT == table5 ]]; then
  for s in $SEEDS; do
    for flag in --no_supp_scan --no_scam --no_skip_gate --no_deep_sup "--supp_init s4d"; do
      run --dataset dataset702 --data_root $AMOS   $flag --seed $s
      run --dataset kvasir     --data_root $KVASIR $flag --seed $s
    done
  done
fi

# ── Table 7: sensitivity to K′ ────────────────────────────────────────────────
if [[ $WHAT == all || $WHAT == table7 ]]; then
  for s in $SEEDS; do
    for k in 2 8 16; do
      run --dataset dataset702 --data_root $AMOS   --k_prime $k --seed $s
      run --dataset kvasir     --data_root $KVASIR --k_prime $k --seed $s
    done
  done
fi

$PY summarize.py
