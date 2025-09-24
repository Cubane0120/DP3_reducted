set -euo pipefail

ALG="dp3"
date="0825"
gpu="0"


ENVS=(
    "metaworld_shelf-place"
    "metaworld_disassemble" 
    "metaworld_stick-pull"
    "metaworld_stick-push"
    "metaworld_pick-place-wall"
)

SEED=(
    "0"
    # "1"
    # "2"
)

THD=(
    "0.99"
    "0.95"
    "0.9"
)



LOG_DIR_origin="${date}/eval_logs_origin_${ALG}"
LOG_DIR_reduct="${date}/eval_logs_reduct_${ALG}"


mkdir -p "$LOG_DIR_origin"
mkdir -p "$LOG_DIR_reduct"

for env in "${ENVS[@]}"; do
    for seed in "${SEED[@]}"; do
        echo "▶️ [${env} | ${seed}] 평가 중..."
        bash scripts/train_policy.sh "$ALG" "$env" "$date" "$seed" "$gpu" #&>fin_log_${ALG}.txt
        echo " fin train_policy"
        bash scripts/eval_policy.sh "$ALG" "$env" "$date" "$seed" "$gpu" \
        2>&1 | tee fin_log_${ALG}.txt \
        | grep -E '^(test_mean_score|inference_fps|num_param):' \
        > "${LOG_DIR_origin}/${env}_${date}_seed${seed}.txt"
        echo " fin eval_policy"

        bash scripts/eval_to_collect_data.sh "$ALG" "$env" "$date" "$seed" "$gpu" &>fin_log_${ALG}.txt
        echo " fin collect_data"
        bash scripts/calculate_SVD.sh "$ALG" "$env" "$date" "$seed" "$gpu" &>fin_log_${ALG}.txt                                                               
        echo " fin calculate_SVD"
        for thd in "${THD[@]}"; do
            bash scripts/train_policy_with_reduction.sh "$ALG" "$env" "$date" "$seed" "$gpu" "$thd" &>fin_log_${ALG}.txt
            echo "  fin train_policy with thd ${thd}"
            bash scripts/eval_policy_with_reduction.sh "$ALG" "$env" "$date" "$seed" "$gpu" "$thd"\
            2>&1 | tee fin_log_${ALG}.txt \
            | grep -E '^(test_mean_score|inference_fps|k_h1|k_h2|num_param):' \
            > "${LOG_DIR_reduct}/${env}_${date}_seed${seed}_thd${thd}.txt"

            echo "  fin eval_policy with thd ${thd}"
        done

        outputs_dir="3D-Diffusion-Policy/data/outputs/${env}-${ALG}-${date}_seed${seed}"
        rm -rf ${outputs_dir}/checkpoints*
        rm -rf ${outputs_dir}/basis
        # rm -rf ${outputs_dir}/collect_data

        echo " fin remove checkpoints and etc"
        echo "✅ 완료: LOG_DIR/${env}_${date}_seed${seed}_thdN.txt"


    done
done

echo "🎉 모든 평가 완료!"