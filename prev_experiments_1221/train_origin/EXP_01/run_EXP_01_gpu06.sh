set -euo pipefail

ALG="dp3"
date="1221"
gpu="6"

exp_type="origin"

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


