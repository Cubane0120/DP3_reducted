set -euo pipefail

ALG="flowpolicy"
date="0813"
gpu="2"


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

LOG_DIR_origin="./eval_logs_origin_${ALG}"
LOG_DIR_reduct="./eval_logs_reduct_${ALG}"


mkdir -p "$LOG_DIR_origin"
mkdir -p "$LOG_DIR_reduct"

for env in "${ENVS[@]}"; do
    for seed in "${SEED[@]}"; do
        echo "▶️ [${env} | ${seed}] 평가 중..."
        bash scripts/train_policy.sh "$ALG" "$env" "$date" "$seed" "$gpu" &>/dev/null
        echo " fin train_policy"
        bash scripts/eval_policy.sh "$ALG" "$env" "$date" "$seed" "$gpu" \
          | grep -E '^(test_mean_score|inference_fps):' \
          > "${LOG_DIR_origin}/${env}_${date}_seed${seed}.txt" 2>/dev/null
        echo " fin eval_policy"

        bash scripts/eval_to_collect_data.sh "$ALG" "$env" "$date" "$seed" "$gpu" &>/dev/null
        echo " fin collect_data"
        bash scripts/calculate_SVD.sh "$ALG" "$env" "$date" "$seed" "$gpu" &>/dev/null                                                                     
        echo " fin calculate_SVD"
        for thd in "${THD[@]}"; do
            bash scripts/train_policy_with_reduction.sh "$ALG" "$env" "$date" "$seed" "$gpu" "$thd" &>/dev/null
            echo "  fin train_policy with thd ${thd}"
            bash scripts/eval_policy_with_reduction.sh "$ALG" "$env" "$date" "$seed" "$gpu" "$thd"\
            | grep -E '^(test_mean_score|inference_fps|k_h1|k_h2):' \
            > "${LOG_DIR_reduct}/${env}_${date}_seed${seed}_thd${thd}.txt" 2>/dev/null
            echo "  fin eval_policy with thd ${thd}"
        done
        echo "✅ 완료: LOG_DIR/${env}_${date}_seed${seed}_thdN.txt"

    done
done

echo "🎉 모든 평가 완료!"