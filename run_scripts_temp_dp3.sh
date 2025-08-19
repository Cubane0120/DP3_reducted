bash scripts/train_policy_with_reduction.sh dp3 metaworld_pick-place-wall 0808 0 0 0.95
bash scripts/train_policy_with_reduction.sh dp3 metaworld_pick-place-wall 0808 0 0 0.9

bash scripts/eval_policy_with_reduction.sh dp3 metaworld_pick-place-wall 0808 0 0 0.99 | grep -E '^(test_mean_score|inference_fps|k_h1|k_h2):' \
            > "./eval_logs_reduct_dp3/metaworld_pick-place-wall_0808_seed0_thd0.99.txt"
            
bash scripts/eval_policy_with_reduction.sh dp3 metaworld_pick-place-wall 0808 0 0 0.95 | grep -E '^(test_mean_score|inference_fps|k_h1|k_h2):' \
            > "./eval_logs_reduct_dp3/metaworld_pick-place-wall_0808_seed0_thd0.95.txt"
bash scripts/eval_policy_with_reduction.sh dp3 metaworld_pick-place-wall 0808 0 0 0.9 | grep -E '^(test_mean_score|inference_fps|k_h1|k_h2):' \
            > "./eval_logs_reduct_dp3/dp3_0808_seed0_thd0.9.txt"
            