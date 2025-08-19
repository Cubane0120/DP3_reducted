bash scripts/eval_policy_with_reduction.sh flowpolicy metaworld_stick-push 0808 0 1 0.99 | grep -E '^(test_mean_score|inference_fps|k_h1|k_h2):' \
            > "./eval_logs_reduct_flowpolicy/metaworld_stick-push_0808_seed0_thd.99.txt"

bash scripts/train_policy_with_reduction.sh flowpolicy metaworld_stick-push 0808 0 1 0.9

bash scripts/eval_policy_with_reduction.sh flowpolicy metaworld_stick-push 0808 0 1 0.9 | grep -E '^(test_mean_score|inference_fps|k_h1|k_h2):' \
            > "./eval_logs_reduct_flowpolicy/metaworld_stick-push_0808_seed0_thd.90.txt"