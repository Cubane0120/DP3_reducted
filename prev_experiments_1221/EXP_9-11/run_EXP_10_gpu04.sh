set -euo pipefail

ALG="fp"
date="1221"
gpu="4"

exp_type="projection_train-oe"

ENVS=(
    # "metaworld_shelf-place"
    # "metaworld_disassemble" 
    "metaworld_assembly"
    "metaworld_hand-insert"
    # "metaworld_basketball"
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


