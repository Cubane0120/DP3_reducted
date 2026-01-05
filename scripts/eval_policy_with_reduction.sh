# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh dp3 adroit_hammer 0322 0 0


DEBUG=False

alg_name=${1}
exp_type=${2}
task_name=${3}
config_name=${alg_name}_${exp_type}
addition_info=${4}
seed=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${6}

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
                            policy_reducted.path_basis=${run_dir}/basis \
                            sub_logging_dir_name=${exp_type} \