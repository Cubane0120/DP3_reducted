set -euo pipefail

ALG="fp"
date="1221"
gpu="0"

exp_type="origin"

ENVS=(
    #very hard
    "metaworld_shelf-place"
    "metaworld_disassemble" 
    "metaworld_assembly"
    "metaworld_hand-insert"
    "metaworld_basketball"
)

SEED=(
    "0"
    # "1"
    # "2"
)

SAMPLE=(
    # "uniform"
    # "2-anchor"
    # "1-anchor"
    "hybrid"
)

LOG_DIR="${date}/eval_logs_${ALG}_${exp_type}"

mkdir -p "$LOG_DIR"

for env in "${ENVS[@]}"; do
    for seed in "${SEED[@]}"; do
        echo "▶️ [${env} | ${seed}] 평가 중..."
        outputs_dir="3D-Diffusion-Policy/data/outputs/${env}-${ALG}-${date}_seed${seed}"

        for sample in "${SAMPLE[@]}"; do
            bash scripts/eval_to_collect_data.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" "$sample" &>fin_log_${ALG}_${exp_type}_${date}_${gpu}.txt
            echo " fin collect_data"

            bash scripts/calculate_SVD.sh "$ALG" "$exp_type" "$env" "$date" "$seed" "$gpu" "$sample" &>fin_log_${ALG}_${exp_type}_${date}_${gpu}.txt
            echo " fin calculate_SVD"
            rm -rf ${outputs_dir}/collect_data
        done
    done
done

echo "🎉 모든 평가 완료!"