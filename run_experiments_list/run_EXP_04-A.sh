set -euo pipefail

ALG="dp3"
date="1221"
gpu="0"

exp_type="fixed_A"

ENVS=(
    #very hard
    "metaworld_shelf-place"
    "metaworld_disassemble" 
    "metaworld_stick-pull"
    "metaworld_stick-push"
    "metaworld_pick-place-wall"

    #hard
    "metaworld_assembly"
    "metaworld_hand-insert"
    "metaworld_pick-out-of-hole"
    "metaworld_pick-place"
    "metaworld_push"


    #medium
    "metaworld_basketball"
    "metaworld_bin-picking"
    "metaworld_box-close"
    "metaworld_coffee-pull"
    "metaworld_coffee-push"
    "metaworld_hammer"
    "metaworld_peg-insert-side"
    "metaworld_push-wall"
    "metaworld_soccer"
    "metaworld_sweep"
    "metaworld_sweep-into"

    #eash
    "metaworld_button-press"
    "metaworld_button-press-topdown"
    "metaworld_button-press-topdown-wall"
    "metaworld_button-press-wall"
    "metaworld_coffee-button"
    "metaworld_dial-turn"
    "metaworld_door-close"
    "metaworld_door-lock"
    "metaworld_door-open"
    "metaworld_door-unlock"
    "metaworld_drawer-close"
    "metaworld_drawer-open"
    "metaworld_faucet-close"
    "metaworld_faucet-open"
    "metaworld_handle-press"
    "metaworld_handle-pull"
    "metaworld_handle-press-side"
    "metaworld_handle-pull-side"
    "metaworld_lever-pull"
    "metaworld_plate-slide"
    "metaworld_plate-slide-back"
    "metaworld_plate-slide-back-side"
    "metaworld_plate-slide-side"
    "metaworld_reach"
    "metaworld_reach-wall"
    "metaworld_window-close"
    "metaworld_window-open"
    "metaworld_peg-unplug-side"
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
        # bash scripts/train_policy_with_reduction.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" &>fin_log_${ALG}_${date}.txt
        # echo " fin train_policy with reduction"
        bash scripts/eval_policy_with_reduction.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" \
        2>&1 | tee fin_log_${ALG}_${date}.txt \
        | grep -E '^(test_mean_score|inference_fps|k_h1|k_h2|num_param):' \
        > "${LOG_DIR}/${env}_seed${seed}.txt"
        echo " fin eval_policy with reduction"

    done
done

echo "🎉 모든 평가 완료!"