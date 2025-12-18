set -euo pipefail


TASK=(
    # "button-press"
    # "button-press-topdown"
    # "button-press-topdown-wall"
    # "button-press-wall"
    # "coffee-button"
    # "dial-turn"
    # "door-close"
    # "door-lock"
    # "door-open"
    # "door-unlock"
    # "drawer-close"
    # "drawer-open"
    # "faucet-close"
    # "faucet-open"
    # "handle-press"
    # "handle-pull"
    # "handle-press-side"
    # "handle-pull-side"
    # "lever-pull"
    # "plate-slide"
    # "plate-slide-back"
    # "plate-slide-back-side"
    # "plate-slide-side"
    # "reach"
    # "reach-wall"
    # "window-close"
    # "window-open"
    # "peg-unplug-side"
    # "basketball"
    # "bin-picking"
    # "box-close"
    # "coffee-pull"
    # "coffee-push"
    # "hammer"
    # "peg-insert-side"
    # "push-wall"
    # "soccer"
    # "sweep"
    # "sweep-into"
    # "assembly"
    # "hand-insert"
    # "pick-out-of-hole"
    # "pick-place"
    # "push"
    "push-back"
)


for task in "${TASK[@]}"; do
    bash scripts/gen_demonstration_metaworld.sh "$task" &>gen_demo_log.txt
    echo "fin ${task}"

done

echo "🎉 모든 평가 완료!"