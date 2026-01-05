set -euo pipefail

ALG="fp"
date="1221"
gpu="3"

exp_type="fixed_A"

ENVS=(
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

    #eash (body)
    "metaworld_handle-pull-side"
    "metaworld_lever-pull"
    "metaworld_plate-slide"
    "metaworld_plate-slide-back"
    "metaworld_plate-slide-back-side"
    "metaworld_plate-slide-side"
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
        bash scripts/train_policy_with_reduction.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" &>fin_log_${ALG}_${exp_type}_${date}_${gpu}.txt
        echo " fin train_policy with reduction"
        bash scripts/eval_policy_with_reduction.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" \
        2>&1 | tee fin_log_${ALG}_${exp_type}_${date}_${gpu}.txt \
        | grep -E '^(test_mean_score|inference_fps|k_1|k_2|P_S1|P_S2|num_param):' \
        > "${LOG_DIR}/${env}_seed${seed}.txt"
        echo " fin eval_policy with reduction"

        outputs_dir="3D-Diffusion-Policy/data/outputs/${env}-${ALG}-${date}_seed${seed}"
        rm -rf ${outputs_dir}/checkpoints_${exp_type}
    done
done

echo "🎉 모든 평가 완료!"



