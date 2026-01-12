set -euo pipefail

ALG="dp3"
date="1221"
gpu="7"

exp_type="origin"

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

    #easy (tail)
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
        echo "▶️ [${env} | ${seed}] 평가 중..."
        # bash scripts/train_policy.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" &>fin_log_${ALG}_${exp_type}_${date}_${gpu}.txt
        # echo " fin train_policy"
        # bash scripts/eval_policy.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" \
        # 2>&1 | tee fin_log_${ALG}_${date}.txt \
        # | grep -E '^(test_mean_score|inference_fps|num_param):' \
        # > "${LOG_DIR}/${env}_seed${seed}.txt"
        # echo " fin eval_policy"
        outputs_dir="3D-Diffusion-Policy/data/outputs/${env}-${ALG}-${date}_seed${seed}"
        rm -rf ${outputs_dir}/basis
        rm -rf ${outputs_dir}/fixed_A
        rm -rf ${outputs_dir}/calculate*
        rm -rf ${outputs_dir}/eval_with_*
        rm -rf ${outputs_dir}/train_with_*
        
        bash scripts/eval_to_collect_data.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" &>fin_log_${ALG}_${exp_type}_${date}_${gpu}.txt
        echo " fin collect_data"

        bash scripts/calculate_SVD.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" &>fin_log_${ALG}_${exp_type}_${date}_${gpu}.txt
        echo " fin calculate_SVD"
        rm -rf ${outputs_dir}/collect_data
    done
done

echo "🎉 모든 평가 완료!"


