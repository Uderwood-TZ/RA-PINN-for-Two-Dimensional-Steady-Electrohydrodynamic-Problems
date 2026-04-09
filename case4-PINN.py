

import os
import csv
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim




@dataclass
class Config:

    seed: int = 42


    use_cuda: bool = True


    nu: float = 0.01
    alpha: float = 0.01
    D: float = 0.01


    A1: float = 1.0
    B1: float = 0.5
    C1: float = 0.3

    A2: float = 0.8
    B2: float = 0.6
    C2: float = 0.4

    A3: float = 1.2
    B3: float = 0.9
    C3: float = 0.5

    A4: float = 1.5
    B4: float = 0.7
    C4: float = 0.6

    A5: float = 1.0
    B5: float = 0.8
    C5: float = 0.3


    x_min: float = 0.0
    x_max: float = 1.0


    n_data_samples: int = 8000
    n_collocation: int = 12000
    n_boundary_each: int = 800


    train_ratio: float = 0.7


    data_batch_size: int = 1024
    bc_batch_size: int = 1024
    pde_batch_size: int = 64
    pde_points_per_epoch: int = 1024
    val_pde_points: int = 1024


    exact_chunk_size: int = 2048
    source_chunk_size: int = 256
    predict_chunk_size: int = 4096


    input_dim: int = 2
    output_dim: int = 5
    hidden_dim: int = 128
    num_hidden_layers: int = 8
    activation: str = "tanh"
    dropout: float = 0.0


    epochs: int = 5000
    lr: float = 1e-3
    weight_decay: float = 1e-8
    print_every: int = 100
    grad_clip: float = 10.0


    scheduler_step_size: int = 1000
    scheduler_gamma: float = 0.7


    use_lbfgs: bool = False
    lbfgs_max_iter: int = 200
    lbfgs_lr: float = 1.0


    lambda_pde: float = 1.0
    lambda_data: float = 10.0
    lambda_bc: float = 10.0


    plot_nx: int = 220
    plot_ny: int = 220
    dpi: int = 300


    results_dir: str = "results"
    empty_cache_every: int = 50


CFG = Config()




def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(use_cuda: bool = True) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dirs(base_dir: str) -> Dict[str, str]:
    dirs = {
        "base": base_dir,
        "figures": os.path.join(base_dir, "figures"),
        "txt": os.path.join(base_dir, "txt"),
        "logs": os.path.join(base_dir, "logs"),
        "models": os.path.join(base_dir, "models"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def to_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device, requires_grad=requires_grad)


def xavier_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)


def batch_iterator(n_samples: int, batch_size: int, shuffle: bool = False, seed: int = 42):
    idx = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield idx[start:end]


class Logger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("Training Log\n")
            f.write("=" * 120 + "\n")

    def log(self, msg: str) -> None:
        print(msg)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(msg + "\n")




def y_bottom_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.10 + 0.05 * torch.sin(2.0 * math.pi * x)


def y_top_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.90 - 0.12 * torch.sin(2.0 * math.pi * x + 0.3)


def y_bottom_numpy(x: np.ndarray) -> np.ndarray:
    return 0.10 + 0.05 * np.sin(2.0 * np.pi * x)


def y_top_numpy(x: np.ndarray) -> np.ndarray:
    return 0.90 - 0.12 * np.sin(2.0 * np.pi * x + 0.3)


def in_domain_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    yb = y_bottom_numpy(x)
    yt = y_top_numpy(x)
    return (y >= yb) & (y <= yt)




def safe_grad(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    grad_outputs: torch.Tensor = None,
    create_graph: bool = True,
    retain_graph: bool = True
) -> torch.Tensor:
    if grad_outputs is None:
        grad_outputs = torch.ones_like(outputs)

    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True
    )[0]

    if grad is None:
        grad = torch.zeros_like(inputs)
    return grad


def gradients(field: torch.Tensor, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad = safe_grad(field, xy, create_graph=True, retain_graph=True)
    return grad[:, 0:1], grad[:, 1:2]


def laplacian(field: torch.Tensor, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fx, fy = gradients(field, xy)
    fxx = safe_grad(fx, xy, create_graph=True, retain_graph=True)[:, 0:1]
    fyy = safe_grad(fy, xy, create_graph=True, retain_graph=True)[:, 1:2]
    return fxx, fyy, fxx + fyy




def u_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:
    pi = math.pi
    return (
        cfg.A1 * torch.sin(pi * x) * torch.cos(pi * y)
        + cfg.B1 * torch.sin(2.0 * pi * x) * torch.sin(2.0 * pi * y)
        + cfg.C1 * torch.exp(-(x ** 2 + y ** 2))
    )


def v_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:
    pi = math.pi
    return (
        cfg.A2 * torch.cos(pi * x) * torch.sin(pi * y)
        + cfg.B2 * torch.sin(2.0 * pi * x) * torch.cos(2.0 * pi * y)
        + cfg.C2 * torch.exp(-(x ** 2 + y ** 2))
    )


def T_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:
    pi = math.pi
    return (
        cfg.A3 * torch.cos(pi * x) * torch.sin(pi * y)
        + cfg.B3 * torch.sin(2.0 * pi * x) * torch.cos(2.0 * pi * y)
        + cfg.C3 * torch.exp(-(x ** 2 + y ** 2))
    )


def p_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:
    pi = math.pi
    return (
        cfg.A4 * torch.sin(pi * x) * torch.cos(pi * y)
        + cfg.B4 * torch.sin(2.0 * pi * x) * torch.sin(2.0 * pi * y)
        + cfg.C4 * torch.exp(-(x ** 2 + y ** 2))
    )


def c_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:
    pi = math.pi
    return (
        cfg.A5 * torch.sin(pi * x) * torch.cos(pi * y)
        + cfg.B5 * torch.sin(2.0 * pi * x) * torch.cos(2.0 * pi * y)
        + cfg.C5 * torch.exp(-(x ** 2 + y ** 2))
    )


def exact_solution(xy: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:
    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    x = xy[:, 0:1]
    y = xy[:, 1:2]

    return {
        "u": u_exact_formula(x, y, cfg),
        "v": v_exact_formula(x, y, cfg),
        "T": T_exact_formula(x, y, cfg),
        "p": p_exact_formula(x, y, cfg),
        "c": c_exact_formula(x, y, cfg),
    }




def compute_source_terms(xy: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:

    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    sol = exact_solution(xy, cfg)
    u, v, T, p, c = sol["u"], sol["v"], sol["T"], sol["p"], sol["c"]

    u_x, u_y = gradients(u, xy)
    v_x, v_y = gradients(v, xy)
    T_x, T_y = gradients(T, xy)
    p_x, p_y = gradients(p, xy)
    c_x, c_y = gradients(c, xy)

    _, _, lap_u = laplacian(u, xy)
    _, _, lap_v = laplacian(v, xy)
    _, _, lap_T = laplacian(T, xy)
    _, _, lap_c = laplacian(c, xy)

    f_mass = u_x + v_y
    f_u = u * u_x + v * u_y + p_x - cfg.nu * lap_u
    f_v = u * v_x + v * v_y + p_y - cfg.nu * lap_v
    f_T = u * T_x + v * T_y - cfg.alpha * lap_T
    f_c = u * c_x + v * c_y - cfg.D * lap_c

    return {
        "f_mass": f_mass,
        "f_u": f_u,
        "f_v": f_v,
        "f_T": f_T,
        "f_c": f_c,
    }




def sample_interior_points(n: int, cfg: Config, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    x = rng.uniform(cfg.x_min, cfg.x_max, size=(n, 1)).astype(np.float32)
    s = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)

    yb = y_bottom_numpy(x)
    yt = y_top_numpy(x)
    y = yb + s * (yt - yb)

    return np.hstack([x, y]).astype(np.float32)


def sample_boundary_points(cfg: Config, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = cfg.n_boundary_each

    # Left boundary x = 0
    x_left = np.zeros((n, 1), dtype=np.float32)
    yb_left = y_bottom_numpy(x_left)
    yt_left = y_top_numpy(x_left)
    s_left = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    y_left = yb_left + s_left * (yt_left - yb_left)
    left = np.hstack([x_left, y_left]).astype(np.float32)

    # Right boundary x = 1
    x_right = np.ones((n, 1), dtype=np.float32)
    yb_right = y_bottom_numpy(x_right)
    yt_right = y_top_numpy(x_right)
    s_right = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    y_right = yb_right + s_right * (yt_right - yb_right)
    right = np.hstack([x_right, y_right]).astype(np.float32)

    # Bottom boundary
    x_bottom = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    y_bottom = y_bottom_numpy(x_bottom).astype(np.float32)
    bottom = np.hstack([x_bottom, y_bottom]).astype(np.float32)

    # Top boundary
    x_top = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    y_top = y_top_numpy(x_top).astype(np.float32)
    top = np.hstack([x_top, y_top]).astype(np.float32)

    return {
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "all": np.vstack([left, right, bottom, top]).astype(np.float32),
    }


def exact_solution_on_points_numpy(
    xy_np: np.ndarray,
    cfg: Config,
    device: torch.device,
    chunk_size: int = 2048
) -> np.ndarray:
    out = []
    for start in range(0, xy_np.shape[0], chunk_size):
        end = min(start + chunk_size, xy_np.shape[0])
        xy_chunk = to_tensor(xy_np[start:end], device=device, requires_grad=True)
        sol = exact_solution(xy_chunk, cfg)
        vals = torch.cat([sol["u"], sol["v"], sol["T"], sol["p"], sol["c"]], dim=1).detach()
        out.append(vals.cpu().numpy())
        del xy_chunk, sol, vals
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.vstack(out)


def source_terms_on_points_numpy(
    xy_np: np.ndarray,
    cfg: Config,
    device: torch.device,
    chunk_size: int = 256
) -> Dict[str, np.ndarray]:
    fmass_all, fu_all, fv_all, fT_all, fc_all = [], [], [], [], []
    for start in range(0, xy_np.shape[0], chunk_size):
        end = min(start + chunk_size, xy_np.shape[0])
        xy_chunk = to_tensor(xy_np[start:end], device=device, requires_grad=True)
        src = compute_source_terms(xy_chunk, cfg)
        fmass_all.append(src["f_mass"].detach().cpu().numpy())
        fu_all.append(src["f_u"].detach().cpu().numpy())
        fv_all.append(src["f_v"].detach().cpu().numpy())
        fT_all.append(src["f_T"].detach().cpu().numpy())
        fc_all.append(src["f_c"].detach().cpu().numpy())
        del xy_chunk, src
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "f_mass": np.vstack(fmass_all),
        "f_u": np.vstack(fu_all),
        "f_v": np.vstack(fv_all),
        "f_T": np.vstack(fT_all),
        "f_c": np.vstack(fc_all),
    }


def split_train_val(arr1: np.ndarray, arr2: np.ndarray, ratio: float, seed: int):
    n = arr1.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(ratio * n)
    tr = idx[:n_train]
    va = idx[n_train:]
    return arr1[tr], arr2[tr], arr1[va], arr2[va]


def generate_plot_points(cfg: Config) -> np.ndarray:

    x = np.linspace(cfg.x_min, cfg.x_max, cfg.plot_nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, cfg.plot_ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    mask = in_domain_numpy(X, Y)
    xy_valid = np.stack([X[mask], Y[mask]], axis=1).astype(np.float32)
    return xy_valid


def generate_dataset(cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:

    xy_data = sample_interior_points(cfg.n_data_samples, cfg, seed=cfg.seed + 10)
    uvTpc_data = exact_solution_on_points_numpy(xy_data, cfg, device, cfg.exact_chunk_size)
    xy_data_tr, uvTpc_data_tr, xy_data_va, uvTpc_data_va = split_train_val(
        xy_data, uvTpc_data, cfg.train_ratio, seed=cfg.seed + 11
    )


    xy_col = sample_interior_points(cfg.n_collocation, cfg, seed=cfg.seed + 20)
    src_np = source_terms_on_points_numpy(xy_col, cfg, device, cfg.source_chunk_size)
    src_all = np.hstack([
        src_np["f_mass"], src_np["f_u"], src_np["f_v"], src_np["f_T"], src_np["f_c"]
    ]).astype(np.float32)
    xy_col_tr, src_tr, xy_col_va, src_va = split_train_val(
        xy_col, src_all, cfg.train_ratio, seed=cfg.seed + 21
    )


    boundary_dict = sample_boundary_points(cfg, seed=cfg.seed + 30)
    xy_bc = boundary_dict["all"]
    uvTpc_bc = exact_solution_on_points_numpy(xy_bc, cfg, device, cfg.exact_chunk_size)
    xy_bc_tr, uvTpc_bc_tr, xy_bc_va, uvTpc_bc_va = split_train_val(
        xy_bc, uvTpc_bc, cfg.train_ratio, seed=cfg.seed + 31
    )


    xy_plot = generate_plot_points(cfg)
    uvTpc_plot = exact_solution_on_points_numpy(xy_plot, cfg, device, cfg.exact_chunk_size)

    return {
        "xy_data_train": to_tensor(xy_data_tr, device=device, requires_grad=False),
        "uvTpc_data_train": to_tensor(uvTpc_data_tr, device=device, requires_grad=False),
        "xy_data_val": to_tensor(xy_data_va, device=device, requires_grad=False),
        "uvTpc_data_val": to_tensor(uvTpc_data_va, device=device, requires_grad=False),

        "xy_col_train": to_tensor(xy_col_tr, device=device, requires_grad=False),
        "src_train": to_tensor(src_tr, device=device, requires_grad=False),
        "xy_col_val": to_tensor(xy_col_va, device=device, requires_grad=False),
        "src_val": to_tensor(src_va, device=device, requires_grad=False),

        "xy_bc_train": to_tensor(xy_bc_tr, device=device, requires_grad=False),
        "uvTpc_bc_train": to_tensor(uvTpc_bc_tr, device=device, requires_grad=False),
        "xy_bc_val": to_tensor(xy_bc_va, device=device, requires_grad=False),
        "uvTpc_bc_val": to_tensor(uvTpc_bc_va, device=device, requires_grad=False),

        "xy_plot": to_tensor(xy_plot, device=device, requires_grad=False),
        "uvTpc_plot": to_tensor(uvTpc_plot, device=device, requires_grad=False),
    }




class PINNNet(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        if cfg.activation.lower() == "tanh":
            act_cls = nn.Tanh
        elif cfg.activation.lower() == "relu":
            act_cls = nn.ReLU
        else:
            raise ValueError("Only tanh or relu is supported.")

        layers = [nn.Linear(cfg.input_dim, cfg.hidden_dim), act_cls()]
        for _ in range(cfg.num_hidden_layers - 1):
            layers += [
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                act_cls(),
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
            ]
        layers += [nn.Linear(cfg.hidden_dim, cfg.output_dim)]

        self.net = nn.Sequential(*layers)
        self.apply(xavier_init)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    return PINNNet(cfg).to(device)




def split_src(src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return src[:, 0:1], src[:, 1:2], src[:, 2:3], src[:, 3:4], src[:, 4:5]


def pde_residuals(
    model: nn.Module,
    xy: torch.Tensor,
    f_mass: torch.Tensor,
    f_u: torch.Tensor,
    f_v: torch.Tensor,
    f_T: torch.Tensor,
    f_c: torch.Tensor,
    cfg: Config
) -> Dict[str, torch.Tensor]:

    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    pred = model(xy)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    T = pred[:, 2:3]
    p = pred[:, 3:4]
    c = pred[:, 4:5]

    u_x, u_y = gradients(u, xy)
    v_x, v_y = gradients(v, xy)
    T_x, T_y = gradients(T, xy)
    p_x, p_y = gradients(p, xy)
    c_x, c_y = gradients(c, xy)

    _, _, lap_u = laplacian(u, xy)
    _, _, lap_v = laplacian(v, xy)
    _, _, lap_T = laplacian(T, xy)
    _, _, lap_c = laplacian(c, xy)

    r_mass = u_x + v_y - f_mass
    r_u = u * u_x + v * u_y + p_x - cfg.nu * lap_u - f_u
    r_v = u * v_x + v * v_y + p_y - cfg.nu * lap_v - f_v
    r_T = u * T_x + v * T_y - cfg.alpha * lap_T - f_T
    r_c = u * c_x + v * c_y - cfg.D * lap_c - f_c

    return {
        "mass": r_mass,
        "momentum_u": r_u,
        "momentum_v": r_v,
        "energy": r_T,
        "concentration": r_c,
    }




def mse_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - true) ** 2)


def compute_data_loss_batched(model: nn.Module, xy: torch.Tensor, uvTpc: torch.Tensor,
                              batch_size: int, epoch_seed: int) -> torch.Tensor:
    total = 0.0
    n = xy.shape[0]
    nb = 0
    for idx in batch_iterator(n, batch_size, shuffle=True, seed=epoch_seed):
        pred = model(xy[idx])
        total = total + mse_loss(pred, uvTpc[idx])
        nb += 1
    return total / max(nb, 1)


def compute_bc_loss_batched(model: nn.Module, xy: torch.Tensor, uvTpc: torch.Tensor,
                            batch_size: int) -> torch.Tensor:
    total = 0.0
    n = xy.shape[0]
    nb = 0
    for idx in batch_iterator(n, batch_size, shuffle=False):
        pred = model(xy[idx])
        total = total + mse_loss(pred, uvTpc[idx])
        nb += 1
    return total / max(nb, 1)


def compute_pde_loss_batched(model: nn.Module, xy: torch.Tensor, src: torch.Tensor,
                             cfg: Config, epoch_seed: int, n_use: int) -> Dict[str, torch.Tensor]:
    n_all = xy.shape[0]
    rng = np.random.default_rng(epoch_seed)
    n_use = min(n_use, n_all)
    chosen = rng.choice(n_all, size=n_use, replace=False)

    xy_use = xy[chosen]
    src_use = src[chosen]

    total_mass = 0.0
    total_u = 0.0
    total_v = 0.0
    total_T = 0.0
    total_c = 0.0
    nb = 0

    for idx in batch_iterator(n_use, cfg.pde_batch_size, shuffle=False):
        xy_batch = xy_use[idx].clone().detach().requires_grad_(True)
        f_mass, f_u, f_v, f_T, f_c = split_src(src_use[idx])

        res = pde_residuals(model, xy_batch, f_mass, f_u, f_v, f_T, f_c, cfg)

        loss_mass = torch.mean(res["mass"] ** 2)
        loss_u = torch.mean(res["momentum_u"] ** 2)
        loss_v = torch.mean(res["momentum_v"] ** 2)
        loss_T = torch.mean(res["energy"] ** 2)
        loss_c = torch.mean(res["concentration"] ** 2)

        total_mass = total_mass + loss_mass
        total_u = total_u + loss_u
        total_v = total_v + loss_v
        total_T = total_T + loss_T
        total_c = total_c + loss_c
        nb += 1

        del xy_batch, f_mass, f_u, f_v, f_T, f_c, res, loss_mass, loss_u, loss_v, loss_T, loss_c

    loss_mass = total_mass / max(nb, 1)
    loss_u = total_u / max(nb, 1)
    loss_v = total_v / max(nb, 1)
    loss_T = total_T / max(nb, 1)
    loss_c = total_c / max(nb, 1)
    loss_pde = loss_mass + loss_u + loss_v + loss_T + loss_c

    return {
        "loss_pde": loss_pde,
        "loss_mass": loss_mass,
        "loss_u": loss_u,
        "loss_v": loss_v,
        "loss_T": loss_T,
        "loss_c": loss_c,
    }


def compute_train_losses(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config, epoch: int) -> Dict[str, torch.Tensor]:
    loss_data = compute_data_loss_batched(
        model,
        dataset["xy_data_train"],
        dataset["uvTpc_data_train"],
        batch_size=cfg.data_batch_size,
        epoch_seed=cfg.seed + epoch
    )

    pde_losses = compute_pde_loss_batched(
        model,
        dataset["xy_col_train"],
        dataset["src_train"],
        cfg,
        epoch_seed=cfg.seed + 10000 + epoch,
        n_use=cfg.pde_points_per_epoch
    )

    loss_bc = compute_bc_loss_batched(
        model,
        dataset["xy_bc_train"],
        dataset["uvTpc_bc_train"],
        batch_size=cfg.bc_batch_size
    )

    loss_total = cfg.lambda_pde * pde_losses["loss_pde"] + cfg.lambda_data * loss_data + cfg.lambda_bc * loss_bc

    return {
        "loss_total": loss_total,
        "loss_pde": pde_losses["loss_pde"],
        "loss_data": loss_data,
        "loss_bc": loss_bc,
        "loss_mass": pde_losses["loss_mass"],
        "loss_u": pde_losses["loss_u"],
        "loss_v": pde_losses["loss_v"],
        "loss_T": pde_losses["loss_T"],
        "loss_c": pde_losses["loss_c"],
    }


def compute_val_losses(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config) -> Dict[str, float]:
    model.eval()

    with torch.no_grad():
        pred_data = model(dataset["xy_data_val"])
        loss_data_val = torch.mean((pred_data - dataset["uvTpc_data_val"]) ** 2).item()

        pred_bc = model(dataset["xy_bc_val"])
        loss_bc_val = torch.mean((pred_bc - dataset["uvTpc_bc_val"]) ** 2).item()

    pde_losses = compute_pde_loss_batched(
        model,
        dataset["xy_col_val"],
        dataset["src_val"],
        cfg,
        epoch_seed=cfg.seed + 999999,
        n_use=cfg.val_pde_points
    )

    loss_pde_val = pde_losses["loss_pde"].item()
    total_val = cfg.lambda_pde * loss_pde_val + cfg.lambda_data * loss_data_val + cfg.lambda_bc * loss_bc_val

    return {
        "val_total_loss": total_val,
        "val_pde_loss": loss_pde_val,
        "val_data_loss": loss_data_val,
        "val_bc_loss": loss_bc_val,
    }




def compute_metrics_numpy(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    diff = pred - true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    denom = np.linalg.norm(true.reshape(-1), ord=2)
    l2 = float(np.linalg.norm(diff.reshape(-1), ord=2) / (denom + 1e-14))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "L2": l2}


def predict_in_batches(model: nn.Module, xy: torch.Tensor, batch_size: int) -> np.ndarray:
    preds = []
    n = xy.shape[0]
    model.eval()
    with torch.no_grad():
        for idx in batch_iterator(n, batch_size, shuffle=False):
            preds.append(model(xy[idx]).detach().cpu().numpy())
    return np.vstack(preds)


def evaluate_on_points(model: nn.Module, xy: torch.Tensor, uvTpc_true: torch.Tensor, cfg: Config) -> Dict[str, Dict[str, float]]:
    pred = predict_in_batches(model, xy, batch_size=cfg.predict_chunk_size)
    true = uvTpc_true.detach().cpu().numpy()

    names = ["u", "v", "T", "p", "c"]
    metrics = {}
    all_mse, all_rmse, all_mae, all_l2 = [], [], [], []

    for i, name in enumerate(names):
        m = compute_metrics_numpy(pred[:, i:i+1], true[:, i:i+1])
        metrics[name] = m
        all_mse.append(m["MSE"])
        all_rmse.append(m["RMSE"])
        all_mae.append(m["MAE"])
        all_l2.append(m["L2"])

    metrics["overall_mean"] = {
        "MSE": float(np.mean(all_mse)),
        "RMSE": float(np.mean(all_rmse)),
        "MAE": float(np.mean(all_mae)),
        "L2": float(np.mean(all_l2)),
    }
    return metrics


def save_metrics(metrics: Dict[str, Dict[str, float]], save_prefix: str) -> None:
    txt_path = save_prefix + ".txt"
    csv_path = save_prefix + ".csv"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Metrics\n")
        f.write("=" * 80 + "\n")
        for key, vals in metrics.items():
            f.write(f"{key}\n")
            for mk, mv in vals.items():
                f.write(f"  {mk}: {mv:.10e}\n")
            f.write("-" * 80 + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Variable", "MSE", "RMSE", "MAE", "L2"])
        for key, vals in metrics.items():
            writer.writerow([key, vals["MSE"], vals["RMSE"], vals["MAE"], vals["L2"]])


def print_metrics(title: str, metrics: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for key, vals in metrics.items():
        print(
            f"{key:>12s} | "
            f"MSE={vals['MSE']:.6e} | "
            f"RMSE={vals['RMSE']:.6e} | "
            f"MAE={vals['MAE']:.6e} | "
            f"L2={vals['L2']:.6e}"
        )
    print("=" * 100)




def save_txt_xyz(filepath: str, x: np.ndarray, y: np.ndarray, value: np.ndarray) -> None:
    data = np.column_stack([x.reshape(-1), y.reshape(-1), value.reshape(-1)])
    header = "x y value"
    np.savetxt(filepath, data, fmt="%.8f %.8f %.8f", header=header, comments="")




def plot_channel_outline(ax):
    x_line = np.linspace(0.0, 1.0, 500)
    yb = y_bottom_numpy(x_line)
    yt = y_top_numpy(x_line)
    ax.plot(x_line, yb, "k-", linewidth=1.2)
    ax.plot(x_line, yt, "k-", linewidth=1.2)
    ax.plot([0, 0], [yb[0], yt[0]], "k-", linewidth=1.2)
    ax.plot([1, 1], [yb[-1], yt[-1]], "k-", linewidth=1.2)


def plot_tricontour_field(x: np.ndarray, y: np.ndarray, value: np.ndarray,
                          title: str, save_path: str, dpi: int = 300) -> None:
    plt.figure(figsize=(8, 4.8))
    ax = plt.gca()
    contour = ax.tricontourf(x, y, value, levels=120, cmap="rainbow")
    plt.colorbar(contour)
    plot_channel_outline(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], fig_dir: str, dpi: int = 300) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(history["loss_total"], label="train total")
    plt.plot(history["val_total_loss"], label="val total")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_total_history.png"), dpi=dpi)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(history["loss_pde"], label="train pde")
    plt.plot(history["loss_data"], label="train data")
    plt.plot(history["loss_bc"], label="train bc")
    plt.plot(history["val_pde_loss"], label="val pde")
    plt.plot(history["val_data_loss"], label="val data")
    plt.plot(history["val_bc_loss"], label="val bc")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Components History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_components_history.png"), dpi=dpi)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(history["loss_mass"], label="continuity")
    plt.plot(history["loss_u"], label="x-momentum")
    plt.plot(history["loss_v"], label="y-momentum")
    plt.plot(history["loss_T"], label="energy")
    plt.plot(history["loss_c"], label="concentration")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PDE Component Loss History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pde_component_history.png"), dpi=dpi)
    plt.close()


def plot_all_results(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config, dirs: Dict[str, str]) -> None:
    pred = predict_in_batches(model, dataset["xy_plot"], batch_size=cfg.predict_chunk_size)
    true = dataset["uvTpc_plot"].detach().cpu().numpy()
    xy = dataset["xy_plot"].detach().cpu().numpy()

    x = xy[:, 0]
    y = xy[:, 1]

    names = ["u", "v", "T", "p", "c"]

    for i, name in enumerate(names):
        pred_val = pred[:, i]
        true_val = true[:, i]
        err_val = np.abs(pred_val - true_val)

        plot_tricontour_field(
            x, y, pred_val,
            f"{name} Prediction",
            os.path.join(dirs["figures"], f"{name}_pred.png"),
            dpi=cfg.dpi
        )
        plot_tricontour_field(
            x, y, true_val,
            f"{name} True",
            os.path.join(dirs["figures"], f"{name}_true.png"),
            dpi=cfg.dpi
        )
        plot_tricontour_field(
            x, y, err_val,
            f"{name} Absolute Error",
            os.path.join(dirs["figures"], f"{name}_abs_error.png"),
            dpi=cfg.dpi
        )

        save_txt_xyz(os.path.join(dirs["txt"], f"{name}_pred.txt"), x, y, pred_val)
        save_txt_xyz(os.path.join(dirs["txt"], f"{name}_true.txt"), x, y, true_val)
        save_txt_xyz(os.path.join(dirs["txt"], f"{name}_abs_error.txt"), x, y, err_val)




def compute_supervised_mse_batched(model: nn.Module, xy: torch.Tensor, uvTpc: torch.Tensor,
                                   batch_size: int) -> float:
    total = 0.0
    n = xy.shape[0]
    nb = 0
    model.eval()
    with torch.no_grad():
        for idx in batch_iterator(n, batch_size, shuffle=False):
            pred = model(xy[idx])
            total += torch.mean((pred - uvTpc[idx]) ** 2).item()
            nb += 1
    return total / max(nb, 1)


def train_model(model: nn.Module, dataset: Dict[str, torch.Tensor],
                cfg: Config, dirs: Dict[str, str], device: torch.device) -> Dict[str, List[float]]:
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
    )

    logger = Logger(os.path.join(dirs["logs"], "training_log.txt"))

    best_val_loss = float("inf")
    best_epoch = -1
    best_model_path = os.path.join(dirs["models"], "best_model.pth")
    final_model_path = os.path.join(dirs["models"], "final_model.pth")

    history = {
        "loss_total": [],
        "loss_pde": [],
        "loss_data": [],
        "loss_bc": [],
        "loss_mass": [],
        "loss_u": [],
        "loss_v": [],
        "loss_T": [],
        "loss_c": [],
        "val_total_loss": [],
        "val_pde_loss": [],
        "val_data_loss": [],
        "val_bc_loss": [],
        "train_mse": [],
        "val_mse": [],
        "epoch_time": [],
        "lr": [],
    }

    with open(os.path.join(dirs["logs"], "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger.log(f"Device: {device}")
    logger.log(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.log(f"Supervised points: {cfg.n_data_samples}")
    logger.log(f"Collocation points: {cfg.n_collocation}")
    logger.log(f"Boundary points total: {4 * cfg.n_boundary_each}")
    logger.log(f"Train ratio: {cfg.train_ratio}")
    logger.log(f"Epochs: {cfg.epochs}")
    logger.log(f"Learning rate: {cfg.lr}")
    logger.log("Model: Pure PINN (MLP)")
    logger.log("-" * 120)

    train_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        losses = compute_train_losses(model, dataset, cfg, epoch)
        losses["loss_total"].backward()

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)

        optimizer.step()
        scheduler.step()

        val_losses = compute_val_losses(model, dataset, cfg)

        train_mse = compute_supervised_mse_batched(
            model, dataset["xy_data_train"], dataset["uvTpc_data_train"], cfg.data_batch_size
        )
        val_mse = compute_supervised_mse_batched(
            model, dataset["xy_data_val"], dataset["uvTpc_data_val"], cfg.data_batch_size
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        history["loss_total"].append(losses["loss_total"].item())
        history["loss_pde"].append(losses["loss_pde"].item())
        history["loss_data"].append(losses["loss_data"].item())
        history["loss_bc"].append(losses["loss_bc"].item())
        history["loss_mass"].append(losses["loss_mass"].item())
        history["loss_u"].append(losses["loss_u"].item())
        history["loss_v"].append(losses["loss_v"].item())
        history["loss_T"].append(losses["loss_T"].item())
        history["loss_c"].append(losses["loss_c"].item())
        history["val_total_loss"].append(val_losses["val_total_loss"])
        history["val_pde_loss"].append(val_losses["val_pde_loss"])
        history["val_data_loss"].append(val_losses["val_data_loss"])
        history["val_bc_loss"].append(val_losses["val_bc_loss"])
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["epoch_time"].append(epoch_time)
        history["lr"].append(current_lr)

        if val_losses["val_total_loss"] < best_val_loss:
            best_val_loss = val_losses["val_total_loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        if epoch == 1 or epoch % cfg.print_every == 0 or epoch == cfg.epochs:
            logger.log(
                f"Epoch [{epoch:5d}/{cfg.epochs}] | "
                f"Total={losses['loss_total'].item():.6e} | "
                f"PDE={losses['loss_pde'].item():.6e} | "
                f"Data={losses['loss_data'].item():.6e} | "
                f"BC={losses['loss_bc'].item():.6e} | "
                f"Mass={losses['loss_mass'].item():.6e} | "
                f"Ux={losses['loss_u'].item():.6e} | "
                f"Vy={losses['loss_v'].item():.6e} | "
                f"T={losses['loss_T'].item():.6e} | "
                f"C={losses['loss_c'].item():.6e} | "
                f"ValTotal={val_losses['val_total_loss']:.6e} | "
                f"ValPDE={val_losses['val_pde_loss']:.6e} | "
                f"ValData={val_losses['val_data_loss']:.6e} | "
                f"ValBC={val_losses['val_bc_loss']:.6e} | "
                f"TrainMSE={train_mse:.6e} | "
                f"ValMSE={val_mse:.6e} | "
                f"LR={current_lr:.3e} | "
                f"EpochTime={epoch_time:.3f}s"
            )

        if device.type == "cuda" and (epoch % cfg.empty_cache_every == 0):
            torch.cuda.empty_cache()

    if cfg.use_lbfgs:
        logger.log("Starting optional LBFGS refinement...")
        lbfgs = optim.LBFGS(model.parameters(), lr=cfg.lbfgs_lr, max_iter=cfg.lbfgs_max_iter)

        def closure():
            lbfgs.zero_grad()
            lbfgs_losses = compute_train_losses(model, dataset, cfg, epoch=0)
            lbfgs_losses["loss_total"].backward()
            return lbfgs_losses["loss_total"]

        lbfgs.step(closure)
        val_losses = compute_val_losses(model, dataset, cfg)
        if val_losses["val_total_loss"] < best_val_loss:
            best_val_loss = val_losses["val_total_loss"]
            best_epoch = cfg.epochs
            torch.save(model.state_dict(), best_model_path)

    train_end = time.time()
    total_training_time = train_end - train_start

    torch.save(model.state_dict(), final_model_path)

    history_csv = os.path.join(dirs["logs"], "loss_history.csv")
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "loss_total", "loss_pde", "loss_data", "loss_bc",
            "loss_mass", "loss_u", "loss_v", "loss_T", "loss_c",
            "val_total_loss", "val_pde_loss", "val_data_loss", "val_bc_loss",
            "train_mse", "val_mse", "epoch_time", "lr"
        ])
        for i in range(len(history["loss_total"])):
            writer.writerow([
                i + 1,
                history["loss_total"][i],
                history["loss_pde"][i],
                history["loss_data"][i],
                history["loss_bc"][i],
                history["loss_mass"][i],
                history["loss_u"][i],
                history["loss_v"][i],
                history["loss_T"][i],
                history["loss_c"][i],
                history["val_total_loss"][i],
                history["val_pde_loss"][i],
                history["val_data_loss"][i],
                history["val_bc_loss"][i],
                history["train_mse"][i],
                history["val_mse"][i],
                history["epoch_time"][i],
                history["lr"][i],
            ])

    with open(os.path.join(dirs["logs"], "timing.txt"), "w", encoding="utf-8") as f:
        f.write("Timing Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Training start timestamp: {train_start:.6f}\n")
        f.write(f"Training end timestamp: {train_end:.6f}\n")
        f.write(f"Total training time (s): {total_training_time:.6f}\n")
        f.write(f"Average epoch time (s): {np.mean(history['epoch_time']):.6f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation total loss: {best_val_loss:.10e}\n")

    logger.log("=" * 120)
    logger.log(f"Best epoch: {best_epoch}")
    logger.log(f"Best validation total loss: {best_val_loss:.10e}")
    logger.log(f"Total training time (s): {total_training_time:.6f}")
    logger.log(f"Best model saved to: {best_model_path}")
    logger.log(f"Final model saved to: {final_model_path}")
    logger.log("=" * 120)

    history["best_epoch"] = [best_epoch]
    history["best_val_loss"] = [best_val_loss]
    history["training_time_sec"] = [total_training_time]

    return history




def main():
    total_start = time.time()

    set_seed(CFG.seed)
    device = get_device(CFG.use_cuda)
    dirs = make_dirs(CFG.results_dir)

    dataset = generate_dataset(CFG, device)
    model = build_model(CFG, device)

    history = train_model(model, dataset, CFG, dirs, device)

    best_model_path = os.path.join(dirs["models"], "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    train_metrics = evaluate_on_points(model, dataset["xy_data_train"], dataset["uvTpc_data_train"], CFG)
    val_metrics = evaluate_on_points(model, dataset["xy_data_val"], dataset["uvTpc_data_val"], CFG)
    plot_metrics = evaluate_on_points(model, dataset["xy_plot"], dataset["uvTpc_plot"], CFG)

    print_metrics("Training-set Metrics", train_metrics)
    print_metrics("Validation-set Metrics", val_metrics)
    print_metrics("Plot-grid Metrics", plot_metrics)

    save_metrics(train_metrics, os.path.join(dirs["logs"], "metrics_train"))
    save_metrics(val_metrics, os.path.join(dirs["logs"], "metrics_val"))
    save_metrics(plot_metrics, os.path.join(dirs["logs"], "metrics_plot"))

    plot_training_curves(history, dirs["figures"], dpi=CFG.dpi)
    plot_all_results(model, dataset, CFG, dirs)

    total_end = time.time()
    total_runtime = total_end - total_start

    with open(os.path.join(dirs["logs"], "runtime.txt"), "w", encoding="utf-8") as f:
        f.write("Runtime Information\n")
        f.write("=" * 80 + "\n")
        f.write(f"Program start timestamp: {total_start:.6f}\n")
        f.write(f"Program end timestamp: {total_end:.6f}\n")
        f.write(f"Total runtime (s): {total_runtime:.6f}\n")
        f.write(f"Training time (s): {history['training_time_sec'][0]:.6f}\n")
        f.write(f"Best epoch: {history['best_epoch'][0]}\n")
        f.write(f"Best validation total loss: {history['best_val_loss'][0]:.10e}\n")
        f.write(f"Device: {device}\n")
        if device.type == "cuda":
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")

    print("\nAll outputs saved under:", dirs["base"])
    print("Subdirectories:")
    for k, v in dirs.items():
        if k != "base":
            print(f"  {k:8s}: {v}")
    print(f"Total runtime: {total_runtime:.6f} s")


if __name__ == "__main__":
    main()