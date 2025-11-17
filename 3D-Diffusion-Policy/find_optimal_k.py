# benchmark_ab_condunet.py
import os, time, tempfile
import itertools
import numpy as np
import torch
import pandas as pd
import random
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D

"""
# Execute berfore running below code with GUI
sudo nvidia-smi -pm 1
sudo nvidia-smi -c EXCLUSIVE_PROCESS
sudo nvidia-smi -pl 180
nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=1'
nvidia-settings -a '[fan:0]/GPUTargetFanSpeed=100'
sudo cpupower -c all frequency-set -g performance

# Terminate process not used
./clean_process.sh


# Turn-off GUI
sudo systemctl isolate multi-user.target
# Ctrl + alt + F3 => move to TTY

# Run it by bash prompt 
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


# run code
taskset -c 0-6 python 3D-Diffusion-Policy/find_optimal_k.py 
"""

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def make_random_basis(in_ch: int, out_ch: int, ortho: bool = True, seed: int = 0):
    """
    reductor용 basis 행렬 (in_ch x out_ch) 생성.
    ortho=True면 입력공간에 대해 직교기저 비슷하게 만들어줌(속도엔 큰 영향 X).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((in_ch, out_ch)).astype(np.float32)
    if not ortho:
        return A
    # 간단한 정규화(빠른 QR 대체)
    A /= (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)
    return A

def write_basis_files(h1_in, a, h2_in, b, seed=0):
    """
    (in_ch, out_ch) = (h1_in, a), (h2_in, b) 모양의 basis를
    임시 파일로 저장하고 경로 반환.
    """
    tmpdir = tempfile.mkdtemp(prefix="ab_bases_")
    p1 = os.path.join(tmpdir, f"basis_h1_{h1_in}x{a}.npy")
    p2 = os.path.join(tmpdir, f"basis_h2_{h2_in}x{b}.npy")
    np.save(p1, make_random_basis(h1_in, a, seed=seed))
    np.save(p2, make_random_basis(h2_in, b, seed=seed+1))
    return p1, p2, tmpdir




@torch.inference_mode()
def bench_one_model(model, B, T, input_dim, global_cond_dim=None, 
                    n_warmup=200, n_iters=500, inner_repeats = 100, device="cuda"):
    """
    주어진 모델의 단일 forward 시간(평균)을 측정.
    """

    

    model = model.to(device).eval()
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(x, t, local_cond=None, global_cond=g)
        print("fin warmup")
        times_ms = []
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        for _ in range(n_iters):
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_ev.record() if device.type == "cuda" else None
            wall_start = time.time()
        
            for _inner in range(inner_repeats):
                _ = model(x, t, local_cond=None, global_cond=g)

            end_ev.record()
            torch.cuda.synchronize()
            elapsed_ms = start_ev.elapsed_time(end_ev)  # ms (전체 inner_repeats 포함)

            per_iter_ms = elapsed_ms / inner_repeats
            times_ms.append(per_iter_ms)

    def hdi_from_samples(samples, coverage=0.95):
        """
        샘플 배열로부터 coverage (e.g. 0.95) 를 갖는
        최단 구간(HDI)을 구하고, (low, center, high)를 리턴.
        center는 해당 구간 안의 median.
        """
        arr = np.asarray(samples, dtype=np.float64)
        arr = np.sort(arr)
        n = len(arr)
        if n == 0:
            return np.nan, np.nan, np.nan

        k = int(np.ceil(coverage * n))
        if k <= 0 or k > n:
            med = float(np.median(arr))
            return med, med, med

        # 길이 k짜리 윈도우 중 폭이 최소인 구간 찾기
        widths = arr[k-1:] - arr[:n-k+1]
        best_start = int(np.argmin(widths))
        best_end = best_start + k  # exclusive

        low = float(arr[best_start])
        high = float(arr[best_end - 1])
        center = float(np.median(arr[best_start:best_end]))
        mean = float(np.mean(arr[best_start:best_end]))
        return low, center, high, mean

    # trim_ratio = 0.1
    # arr = np.array(times_ms, dtype=np.float64)
    # print(f"per-iter stats: min={arr.min():.4f}, max={arr.max():.4f}, median={np.median(arr):.4f}")
    # if 0.0 < trim_ratio < 0.5:
    #     lo = np.quantile(arr, trim_ratio)
    #     hi = np.quantile(arr, 1 - trim_ratio)
    #     arr = arr[(arr >= lo) & (arr <= hi)]

    # print(f"per-iter stats: min={arr.min():.4f}, max={arr.max():.4f}, median={np.median(arr):.4f}")
    # return float(np.min(arr))
    coverage = 0.96
    arr = np.array(times_ms, dtype=np.float64)
    low, center, high, mean = hdi_from_samples(arr, coverage=coverage)
    print(f"per-iter raw: min={arr.min():.4f}, max={arr.max():.4f}, median={np.median(arr):.4f}, mean={np.mean(arr):.4f}")
    print(f"HDI {coverage*100}%: [{low:.4f}, {high:.4f}], center={center:.4f}, mean={mean:.4f}")
    return float(center), float(low), float(arr.min())


def build_model_with_ab(
    a, b,
    *,
    input_dim,
    global_cond_dim,
    diffusion_step_embed_dim,
    down_dims,        # e.g. [256, 1024, 2048]
    kernel_size,
    n_groups,
    condition_type,
    use_down_condition,
    use_mid_condition,
    use_up_condition,
    collect_data=False,
    collect_data_path=None,
    mamba_version=None
):
    """
    a,b에 맞는 basis 파일을 임시 생성하여 모델을 빌드.
    reductor_h1: in_ch = down_dims[1]
    reductor_h2: in_ch = down_dims[2]
    """
    h1_in = down_dims[1]
    h2_in = down_dims[2]

    path_basis_h1, path_basis_h2, tmpdir = write_basis_files(h1_in, a, h2_in, b)
        
    model = ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
        condition_type=condition_type,
        use_down_condition=use_down_condition,
        use_mid_condition=use_mid_condition,
        use_up_condition=use_up_condition,
        collect_data=collect_data,
        collect_data_path=collect_data_path,
        path_basis_h1=path_basis_h1,
        path_basis_h2=path_basis_h2,
        mamba_version=mamba_version,
    )
    # tmpdir 경로를 모델에 붙여서 나중에 정리할 수 있게(선택)
    model._tmpdir_for_bases = None
    return model

def build_model_without_ab(
    *,
    input_dim,
    global_cond_dim,
    diffusion_step_embed_dim,
    down_dims,        # e.g. [256, 1024, 2048]
    kernel_size,
    n_groups,
    condition_type,
    use_down_condition,
    use_mid_condition,
    use_up_condition,
    collect_data=False,
    collect_data_path=None,
    mamba_version=None
):
    """
    a,b에 맞는 basis 파일을 임시 생성하여 모델을 빌드.
    reductor_h1: in_ch = down_dims[1]
    reductor_h2: in_ch = down_dims[2]
    """
        
    model = ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
        condition_type=condition_type,
        use_down_condition=use_down_condition,
        use_mid_condition=use_mid_condition,
        use_up_condition=use_up_condition,
        collect_data=collect_data,
        collect_data_path=collect_data_path,
        mamba_version=mamba_version,
    )
    # tmpdir 경로를 모델에 붙여서 나중에 정리할 수 있게(선택)
    model._tmpdir_for_bases = None
    return model

def benchmark_grid(
    base_configs,
    input_dim,
    global_cond_dim,
    diffusion_step_embed_dim,
    down_dims,             # 예: [256, 1024, 2048]
    kernel_size=3,
    n_groups=8,
    condition_type="film",
    use_down_condition=True,
    use_mid_condition=True,
    use_up_condition=True,
    collect_data=False,
    collect_data_path=None,
    mamba_version=None,
    # 벤치마크 설정
    B=1, T=8,
    n_warmup=50, n_iters=100, inner_repeats=100,
    device="cuda",
    reps=3,
):
    if base_configs is not None:
        configs = []
        # print(base_configs)
        for rep in range(reps):
            for (a, b) in base_configs:
                configs.append({"a": a, "b": b, "rep": rep})
        # print(configs)
        # 순서 섞어서 thermal drift 영향 골고루 섞기
        random.shuffle(configs)

        rows = []
        for index, cfg in enumerate(configs):
            a = cfg["a"]
            b = cfg["b"]
            rep = cfg["rep"]
            model = build_model_with_ab(
                a, b,
                input_dim=input_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
                use_down_condition=use_down_condition,
                use_mid_condition=use_mid_condition,
                use_up_condition=use_up_condition,
                collect_data=collect_data,
                collect_data_path=collect_data_path,
                mamba_version=mamba_version,
            )
            try:
                ms, low, min = bench_one_model(
                    model, B=B, T=T, input_dim=input_dim,
                    global_cond_dim=global_cond_dim,
                    n_warmup=n_warmup, n_iters=n_iters, inner_repeats=inner_repeats,
                    device=device
                )
                rows.append({"a": a, "b": b, "rep": rep, "ms_per_iter": ms, "ms_per_iter_low": low, "ms_per_iter_min": min})
            finally:
                # 임시 basis 디렉터리 정리
                tmpdir = getattr(model, "_tmpdir_for_bases", None)
                del model
                if tmpdir and os.path.isdir(tmpdir):
                    # 안전하게 파일 삭제
                    for f in os.listdir(tmpdir):
                        try:
                            os.remove(os.path.join(tmpdir, f))
                        except:
                            pass
                    try:
                        os.rmdir(tmpdir)
                    except:
                        pass
                
                print(f"current model's index : {index}, total num ={len(configs)}")

        df = pd.DataFrame(rows).sort_values(["a", "b","rep"]).reset_index(drop=True)
        return df
    else:
        configs = []
        for rep in range(reps):
            configs.append({"rep": rep})

        rows = []
        for index, cfg in enumerate(configs):
            rep = cfg["rep"]
            model = build_model_without_ab(
                input_dim=input_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
                use_down_condition=use_down_condition,
                use_mid_condition=use_mid_condition,
                use_up_condition=use_up_condition,
                collect_data=collect_data,
                collect_data_path=collect_data_path,
                mamba_version=mamba_version,
            )
            try:
                ms, low, min = bench_one_model(
                    model, B=B, T=T, input_dim=input_dim,
                    global_cond_dim=global_cond_dim,
                    n_warmup=n_warmup, n_iters=n_iters, inner_repeats=inner_repeats,
                    device=device
                )
                rows.append({"rep": rep, "ms_per_iter": ms, "ms_per_iter_low": low, "ms_per_iter_min": min})
            finally:
                # 임시 basis 디렉터리 정리
                tmpdir = getattr(model, "_tmpdir_for_bases", None)
                del model
                if tmpdir and os.path.isdir(tmpdir):
                    # 안전하게 파일 삭제
                    for f in os.listdir(tmpdir):
                        try:
                            os.remove(os.path.join(tmpdir, f))
                        except:
                            pass
                    try:
                        os.rmdir(tmpdir)
                    except:
                        pass
                
                print(f"current model's index : {index}, total num ={len(configs)}")

        df = pd.DataFrame(rows).sort_values(["rep"]).reset_index(drop=True)
        return df



if __name__ == "__main__":
    # ===== 예시 파라미터 (당신 프로젝트 값으로 바꾸세요) =====
    input_dim = 4           # x의 채널(헤더 블록 출력 채널)
    global_cond_dim = 256     # 없으면 None
    diffusion_step_embed_dim = 128
    down_dims = [512, 1024, 2048]   # 코드 구조에 맞추어 사용 중인 값
    kernel_size = 5
    n_groups = 8
    condition_type = "film"
    use_down_condition = True
    use_mid_condition = True
    use_up_condition = True
    mamba_version = None      # "mambavision_v1" 등을 쓰는 경우 지정

    # 벤치 설정
    B, T = 1, 4               # 실제 시나리오에 맞게


    # a_list = [val for val in range(16, 256 + 1, 16)]
    # b_list = [val for val in range(32, 512 + 1, 32)]
    # a_list = [64, 128]
    # b_list = [128, 256]
    # base_configs = list(itertools.product(a_list, b_list))

    base_configs = [(8*i, 16*i) for i in range(1,13)]

    # base_configs = None
    # print(base_configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global x, t, g
    x = torch.randn(B, T, input_dim, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    g = torch.randn(B, global_cond_dim, device=device) if global_cond_dim is not None else None

    #warmup
    _ = benchmark_grid(
        [(128, 256)],
        input_dim=input_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
        condition_type=condition_type,
        use_down_condition=use_down_condition,
        use_mid_condition=use_mid_condition,
        use_up_condition=use_up_condition,
        mamba_version=mamba_version,
        B=B, T=T,
        n_warmup=5000, n_iters=20, inner_repeats=1000,
        #  n_warmup=5, n_iters=2, inner_repeats=1,
        device=device,
        reps=1
    )

    #real run
    # df = benchmark_grid(
    #     a_list, b_list,
    #     input_dim=input_dim,
    #     global_cond_dim=global_cond_dim,
    #     diffusion_step_embed_dim=diffusion_step_embed_dim,
    #     down_dims=down_dims,
    #     kernel_size=kernel_size,
    #     n_groups=n_groups,
    #     condition_type=condition_type,
    #     use_down_condition=use_down_condition,
    #     use_mid_condition=use_mid_condition,
    #     use_up_condition=use_up_condition,
    #     mamba_version=mamba_version,
    #     B=B, T=T,
    #     n_warmup=500, n_iters=200, inner_repeats=10,
    #     device=device,
    #     reps=5
    # )
    df = benchmark_grid(
        base_configs,
        input_dim=input_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
        condition_type=condition_type,
        use_down_condition=use_down_condition,
        use_mid_condition=use_mid_condition,
        use_up_condition=use_up_condition,
        mamba_version=mamba_version,
        B=B, T=T,
        n_warmup=1000, n_iters=300, inner_repeats=30,
        device=device,
        reps=30
    )

    print(df.to_string(index=False))
    
    import os, time
    os.makedirs("bench_out", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    

    # 1) Long-form 원본 저장 (엑셀/CSV)
    raw_csv = f"bench_out/ab_bench_raw_{stamp}.csv"
    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")


    # # 2) 2D 피벗(행=b, 열=a, 값=ms_per_iter)
    # pivot = (
    #     df.groupby(["b", "a"])["ms_per_iter"]
    #     .median()
    #     .round(10)
    #     .unstack("a")
    #     .sort_index()
    #     .sort_index(axis=1)
    # )
    # pivot_csv = f"bench_out/ab_bench_pivot_ms_per_iter_{stamp}.csv"
    # pivot.to_csv(pivot_csv, encoding="utf-8-sig", float_format="%.3f")

    # print(f"\n[저장됨]\n- raw:   {raw_csv}\n- pivot: {pivot_csv}")
