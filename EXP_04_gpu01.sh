set -euo pipefail

ALG="simple-fp"
date="0107"
gpu="1"

exp_type="origin"

ENVS=(
    # "metaworld_shelf-place"
    # "metaworld_disassemble" 
    # "metaworld_stick-pull"
    # "metaworld_stick-push"
    # "metaworld_pick-place-wall"
    # "metaworld_assembly"
    # "metaworld_hand-insert"
    # "metaworld_pick-out-of-hole"
    # "metaworld_pick-place"
    # "metaworld_push"
    # "metaworld_bin-picking"
    # "metaworld_box-close"
    # "metaworld_coffee-pull"
    # "metaworld_coffee-push"
    # "metaworld_hammer"
    # "metaworld_peg-insert-side"
    # "metaworld_push-wall"
    # "metaworld_soccer"
    # "metaworld_sweep"
    # "metaworld_sweep-into"
    # "metaworld_button-press" ----
    # "metaworld_button-press-topdown"
    # "metaworld_button-press-topdown-wall"
    # "metaworld_button-press-wall"
    # "metaworld_coffee-button"
    # "metaworld_dial-turn"
    # "metaworld_door-close"
    # "metaworld_door-lock"
    # "metaworld_door-open"
    # "metaworld_door-unlock"
    "metaworld_drawer-close"
    "metaworld_drawer-open"
    "metaworld_faucet-close"
    "metaworld_faucet-open"
    "metaworld_handle-press"
    "metaworld_handle-pull"
    "metaworld_handle-press-side"
    "metaworld_handle-pull-side"
    "metaworld_lever-pull"
    # "metaworld_plate-slide"
    # "metaworld_plate-slide-back"
    # "metaworld_plate-slide-back-side"
    # "metaworld_plate-slide-side"
    # "metaworld_reach"
    # "metaworld_reach-wall"
    # "metaworld_window-close"
    # "metaworld_window-open"
    # "metaworld_peg-unplug-side"
)

SEED=(
    "0"
    # "1"
    # "2"
)


LOG_DIR="${date}/eval_logs_${ALG}_${exp_type}"

mkdir -p "$LOG_DIR"

for env in "${ENVS[@]}"; do
    for seed in "${SEED[@]}"; do
        echo "▶️ [${env} | ${seed}] 평가 중..."
        bash scripts/train_policy.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" &>fin_log_${ALG}_${exp_type}_${date}_${gpu}.txt
        echo " fin train_policy"
        bash scripts/eval_policy.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" \
        2>&1 | tee fin_log_${ALG}_${exp_type}_${date}_${gpu}.txt \
        | grep -E '^(test_mean_score|inference_fps|num_param):' \
        > "${LOG_DIR}/${env}_seed${seed}.txt"
        echo " fin eval_policy"
        outputs_dir="3D-Diffusion-Policy/data/outputs/${env}-${ALG}-${date}_seed${seed}"
        rm -rf ${outputs_dir}/checkpoints
    done
done

echo "🎉 모든 평가 완료!"


