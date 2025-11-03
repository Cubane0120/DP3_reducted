# benchmark_ab_condunet.py
import os, time, tempfile
import itertools
import numpy as np
import torch
import pandas as pd
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D

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
                    n_warmup=200, n_iters=100, device="cuda"):
    """
    주어진 모델의 단일 forward 시간(평균)을 측정.
    """
    inner_repeats = 10
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with torch.inference_mode():
        times = [] 
        x = torch.empty(B, T, input_dim, device=device)
        t = torch.empty(B, dtype=torch.long, device=device)
        g = None
        if global_cond_dim is not None:
            g = torch.empty(B, global_cond_dim, device=device)
            
        for _ in range(n_warmup):
            x.normal_()
            t.random_(0, 1000)
            if g is not None:
                g.normal_()
            _ = model(x, t, local_cond=None, global_cond=g)
            
        times_ms = []
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        for _ in range(n_iters):
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_ev.record() if device.type == "cuda" else None
            wall_start = time.time()
        
            for _inner in range(inner_repeats):
                x.normal_()
                t.random_(0, 1000)
                if g is not None:
                    g.normal_()
                _ = model(x, t, local_cond=None, global_cond=g)

            torch.cuda.synchronize() if device.type == "cuda" else None
            end_ev.record()
            torch.cuda.synchronize()
            elapsed_ms = start_ev.elapsed_time(end_ev)  # ms (전체 inner_repeats 포함)

            per_iter_ms = elapsed_ms / inner_repeats
            times_ms.append(per_iter_ms)
            
    trim_ratio = 0.1
    arr = np.array(times_ms, dtype=np.float64)
    if 0.0 < trim_ratio < 0.5:
        lo = np.quantile(arr, trim_ratio)
        hi = np.quantile(arr, 1 - trim_ratio)
        arr = arr[(arr >= lo) & (arr <= hi)]

    return float(np.median(arr))

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
    model._tmpdir_for_bases = tmpdir
    return model

def benchmark_grid(
    a_list, b_list,
    *,
    # 모델 하이퍼 파라미터(당신이 쓰는 값으로 넣으세요)
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
    n_warmup=50, n_iters=100,
    device="cuda",
):
    rows = []
    for a, b in itertools.product(a_list, b_list):
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
            sec = bench_one_model(
                model, B=B, T=T, input_dim=input_dim,
                global_cond_dim=global_cond_dim,
                n_warmup=n_warmup, n_iters=n_iters,
                device=device
            )
            rows.append({"a": a, "b": b, "ms_per_iter": sec})
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

    df = pd.DataFrame(rows).sort_values(["a", "b"]).reset_index(drop=True)
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

    
    a_list = [val for val in range(8, 512 + 1, 8)]
    b_list = [val for val in range(16, 1024 + 1, 8)]

    df = benchmark_grid(
        a_list, b_list,
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
        n_warmup=50, n_iters=100,
        device="cuda",
    )
    print(df.to_string(index=False))
    
    import os, time
    os.makedirs("bench_out", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # 1) Long-form 원본 저장 (엑셀/CSV)
    raw_csv = f"bench_out/ab_bench_raw_{stamp}.csv"
    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")


    # 2) 2D 피벗(행=b, 열=a, 값=ms_per_iter)
    pivot = (
        df.pivot(index="b", columns="a", values="ms_per_iter")
          .sort_index()
          .sort_index(axis=1)
          .round(3)
    )
    pivot_csv = f"bench_out/ab_bench_pivot_ms_per_iter_{stamp}.csv"
    pivot.to_csv(pivot_csv, encoding="utf-8-sig", float_format="%.3f")


    print(f"\n[저장됨]\n- raw:   {raw_csv}\n- pivot: {pivot_csv}")