# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh dp3 adroit_hammer 0322 0 0


DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=${5}
threshold=${6}
whitening=${7}

if [[ "$whitening" == "true" ]]; then
    basis_directory="basis/threshold_${threshold}_whitening"
else
    basis_directory="basis/threshold_${threshold}"
fi

#threshold="0.99" 

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python eval_with_reduction.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training_reducted.debug=$DEBUG \
                            training.seed=${seed} \
                            training_reducted.seed=${seed} \
                            training_reducted.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            policy_reducted.path_basis_h1=${run_dir}/${basis_directory}/latent_h1.npy \
                            policy_reducted.path_basis_h2=${run_dir}/${basis_directory}/latent_h2.npy \
                            threshold=${threshold} \
                            +whitening=${whitening} \