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
                    n_warmup=50, n_iters=100, device="cuda"):
    """
    주어진 모델의 단일 forward 시간(평균)을 측정.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    with torch.inference_mode():
        for _ in range(n_warmup):
            x = torch.randn(B, T, input_dim, device=device)
            # timestep: (B,)
            t = torch.randint(low=0, high=1000, size=(B,), device=device)
            # global_cond: (B, global_cond_dim) or None
            g = None
            if global_cond_dim is not None:
                g = torch.randn(B, global_cond_dim, device=device)
                
            _ = model(x, t, local_cond=None, global_cond=g)
            
        total_time = 0.0
        for _ in range(n_iters):
            x = torch.randn(B, T, input_dim, device=device)
            # timestep: (B,)
            t = torch.randint(low=0, high=1000, size=(B,), device=device)
            # global_cond: (B, global_cond_dim) or None
            g = None
            if global_cond_dim is not None:
                g = torch.randn(B, global_cond_dim, device=device)
                
            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.time()
        
            _ = model(x, t, local_cond=None, global_cond=g)

            torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.time()
            total_time += (end - start)
        
    return total_time / n_iters *1000 # sec/iter

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
            rows.append({"a": a, "b": b, "sec_per_iter": sec})
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
    B, T = 1, 16               # 실제 시나리오에 맞게
    # a_list = [val for val in range(32, 256, 32)]
    # b_list = [val for val in range(32, 512, 32)]
    a_list = [val for val in range(32, 96, 8)]
    b_list = [val for val in range(32, 256, 8)]
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