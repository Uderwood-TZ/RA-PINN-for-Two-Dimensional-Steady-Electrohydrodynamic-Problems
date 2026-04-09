

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
    results_dir: str = "results"


    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0


    nu: float = 0.01
    kappa: float = 0.01
    D: float = 0.01
    alpha_e: float = 0.4
    beta_tc: float = 0.5
    gamma_ct: float = 0.3


    n_f: int = 10000
    n_data: int = 6000
    n_bc_each: int = 600


    train_ratio: float = 0.7


    input_dim: int = 2
    output_dim: int = 5
    hidden_dim: int = 128
    num_blocks: int = 6
    activation: str = "tanh"
    use_learnable_alpha: bool = True
    dropout: float = 0.0


    adam_epochs: int = 5000
    adam_lr: float = 1e-3
    weight_decay: float = 1e-8
    print_every: int = 100
    grad_clip: float = 10.0


    use_scheduler: bool = True
    scheduler_step_size: int = 1000
    scheduler_gamma: float = 0.7


    use_lbfgs: bool = True
    lbfgs_max_iter: int = 300
    lbfgs_lr: float = 1.0


    w_pde: float = 1.0
    w_bc: float = 10.0
    w_data: float = 10.0


    pde_batch_size: int = 256
    data_batch_size: int = 1024
    bc_batch_size: int = 1024
    predict_batch_size: int = 4096
    source_chunk_size: int = 256


    test_nx: int = 200
    test_ny: int = 200
    dpi: int = 300


    empty_cache_every: int = 100


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
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def to_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device, requires_grad=requires_grad)


def xavier_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)


def batch_index_iterator(n: int, batch_size: int, shuffle: bool = False, seed: int = 42):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        yield idx[s:e]


class Logger:
    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("training_log\n")
            f.write("=" * 120 + "\n")

    def log(self, msg: str) -> None:
        print(msg)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(msg + "\n")




def u_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    return torch.sin(pi * x) * torch.cos(pi * y) + 0.1 * torch.sin(2 * pi * x) * torch.cos(3 * pi * y)


def v_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    return torch.cos(pi * x) * torch.sin(pi * y) + 0.1 * torch.cos(2 * pi * x) * torch.sin(3 * pi * y)


def T_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    return torch.sin(pi * x) * torch.sin(pi * y) + 0.2 * torch.cos(pi * x) * torch.cos(pi * y)


def p_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    return torch.cos(pi * x) * torch.cos(pi * y) + 0.2 * torch.sin(2 * pi * x) * torch.sin(2 * pi * y)


def c_exact(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    return torch.cos(pi * x) * torch.sin(pi * y) + 0.15 * torch.cos(2 * pi * x) * torch.cos(3 * pi * y)


def exact_solution(xy: torch.Tensor) -> Dict[str, torch.Tensor]:
    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return {
        "u": u_exact(x, y),
        "v": v_exact(x, y),
        "T": T_exact(x, y),
        "p": p_exact(x, y),
        "c": c_exact(x, y),
    }




def safe_grad(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    grad_outputs: torch.Tensor = None,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    if grad_outputs is None:
        grad_outputs = torch.ones_like(outputs)

    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]

    if grad is None:
        grad = torch.zeros_like(inputs)
    return grad


def gradients(field: torch.Tensor, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g = safe_grad(field, xy, create_graph=True, retain_graph=True)
    return g[:, 0:1], g[:, 1:2]


def laplacian(field: torch.Tensor, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fx, fy = gradients(field, xy)
    fxx = safe_grad(fx, xy, create_graph=True, retain_graph=True)[:, 0:1]
    fyy = safe_grad(fy, xy, create_graph=True, retain_graph=True)[:, 1:2]
    return fxx, fyy, fxx + fyy




def striped_electrode_envelope(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    pi = math.pi
    stripes = 1.0 + 0.35 * torch.sin(6.0 * pi * x) + 0.15 * torch.sin(12.0 * pi * x + 0.4)
    y_mod = 0.8 + 0.2 * torch.cos(2.0 * pi * y)
    return stripes * y_mod


def electric_forcing(u: torch.Tensor, v: torch.Tensor, T: torch.Tensor, c: torch.Tensor,
                     xy: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:

    x = xy[:, 0:1]
    y = xy[:, 1:2]
    env = striped_electrode_envelope(x, y)
    T_x, T_y = gradients(T, xy)
    Fe_x = cfg.alpha_e * env * c * T_x
    Fe_y = cfg.alpha_e * env * c * T_y
    return Fe_x, Fe_y




def compute_source_terms(xy: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:

    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    sol = exact_solution(xy)
    u = sol["u"]
    v = sol["v"]
    T = sol["T"]
    p = sol["p"]
    c = sol["c"]

    u_x, u_y = gradients(u, xy)
    v_x, v_y = gradients(v, xy)
    T_x, T_y = gradients(T, xy)
    p_x, p_y = gradients(p, xy)
    c_x, c_y = gradients(c, xy)

    _, _, lap_u = laplacian(u, xy)
    _, _, lap_v = laplacian(v, xy)
    _, _, lap_T = laplacian(T, xy)
    _, _, lap_c = laplacian(c, xy)

    Fe_x, Fe_y = electric_forcing(u, v, T, c, xy, cfg)

    f_mass = u_x + v_y
    f_u = u * u_x + v * u_y + p_x - cfg.nu * lap_u - Fe_x
    f_v = u * v_x + v * v_y + p_y - cfg.nu * lap_v - Fe_y
    f_T = u * T_x + v * T_y - cfg.kappa * lap_T - cfg.beta_tc * u * c
    f_c = u * c_x + v * c_y - cfg.D * lap_c - cfg.gamma_ct * T

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
    y = rng.uniform(cfg.y_min, cfg.y_max, size=(n, 1)).astype(np.float32)
    return np.hstack([x, y]).astype(np.float32)


def sample_boundary_points(cfg: Config, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = cfg.n_bc_each


    x_left = np.zeros((n, 1), dtype=np.float32)
    y_left = rng.uniform(cfg.y_min, cfg.y_max, size=(n, 1)).astype(np.float32)
    left = np.hstack([x_left, y_left]).astype(np.float32)


    x_right = np.ones((n, 1), dtype=np.float32)
    y_right = rng.uniform(cfg.y_min, cfg.y_max, size=(n, 1)).astype(np.float32)
    right = np.hstack([x_right, y_right]).astype(np.float32)


    x_bottom = rng.uniform(cfg.x_min, cfg.x_max, size=(n, 1)).astype(np.float32)
    y_bottom = np.zeros((n, 1), dtype=np.float32)
    bottom = np.hstack([x_bottom, y_bottom]).astype(np.float32)


    x_top = rng.uniform(cfg.x_min, cfg.x_max, size=(n, 1)).astype(np.float32)
    y_top = np.ones((n, 1), dtype=np.float32)
    top = np.hstack([x_top, y_top]).astype(np.float32)

    return {
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "all": np.vstack([left, right, bottom, top]).astype(np.float32),
    }


def exact_solution_on_points_numpy(xy_np: np.ndarray, device: torch.device, chunk_size: int = 2048) -> np.ndarray:
    outputs = []
    for s in range(0, xy_np.shape[0], chunk_size):
        e = min(s + chunk_size, xy_np.shape[0])
        xy_chunk = to_tensor(xy_np[s:e], device=device, requires_grad=True)
        sol = exact_solution(xy_chunk)
        arr = torch.cat([sol["u"], sol["v"], sol["T"], sol["p"], sol["c"]], dim=1).detach().cpu().numpy()
        outputs.append(arr)
        del xy_chunk, sol, arr
    return np.vstack(outputs)


def source_terms_on_points_numpy(xy_np: np.ndarray, cfg: Config, device: torch.device,
                                 chunk_size: int = 256) -> Dict[str, np.ndarray]:
    f_mass_all, f_u_all, f_v_all, f_T_all, f_c_all = [], [], [], [], []
    for s in range(0, xy_np.shape[0], chunk_size):
        e = min(s + chunk_size, xy_np.shape[0])
        xy_chunk = to_tensor(xy_np[s:e], device=device, requires_grad=True)
        src = compute_source_terms(xy_chunk, cfg)
        f_mass_all.append(src["f_mass"].detach().cpu().numpy())
        f_u_all.append(src["f_u"].detach().cpu().numpy())
        f_v_all.append(src["f_v"].detach().cpu().numpy())
        f_T_all.append(src["f_T"].detach().cpu().numpy())
        f_c_all.append(src["f_c"].detach().cpu().numpy())
        del xy_chunk, src
    return {
        "f_mass": np.vstack(f_mass_all),
        "f_u": np.vstack(f_u_all),
        "f_v": np.vstack(f_v_all),
        "f_T": np.vstack(f_T_all),
        "f_c": np.vstack(f_c_all),
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


def generate_test_grid(cfg: Config) -> np.ndarray:
    xs = np.linspace(cfg.x_min, cfg.x_max, cfg.test_nx, dtype=np.float32)
    ys = np.linspace(cfg.y_min, cfg.y_max, cfg.test_ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)
    return xy


def generate_dataset(cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:

    xy_data = sample_interior_points(cfg.n_data, cfg, seed=cfg.seed + 10)
    uvTpc_data = exact_solution_on_points_numpy(xy_data, device=device, chunk_size=cfg.predict_batch_size)
    xy_data_tr, uvTpc_data_tr, xy_data_va, uvTpc_data_va = split_train_val(
        xy_data, uvTpc_data, cfg.train_ratio, seed=cfg.seed + 11
    )


    xy_f = sample_interior_points(cfg.n_f, cfg, seed=cfg.seed + 20)
    src_np = source_terms_on_points_numpy(xy_f, cfg, device, cfg.source_chunk_size)
    src_all = np.hstack([src_np["f_mass"], src_np["f_u"], src_np["f_v"], src_np["f_T"], src_np["f_c"]]).astype(np.float32)


    bc_dict = sample_boundary_points(cfg, seed=cfg.seed + 30)
    xy_bc = bc_dict["all"]
    uvTpc_bc = exact_solution_on_points_numpy(xy_bc, device=device, chunk_size=cfg.predict_batch_size)


    xy_test = generate_test_grid(cfg)
    uvTpc_test = exact_solution_on_points_numpy(xy_test, device=device, chunk_size=cfg.predict_batch_size)

    return {
        "xy_data_train": to_tensor(xy_data_tr, device=device, requires_grad=False),
        "uvTpc_data_train": to_tensor(uvTpc_data_tr, device=device, requires_grad=False),
        "xy_data_val": to_tensor(xy_data_va, device=device, requires_grad=False),
        "uvTpc_data_val": to_tensor(uvTpc_data_va, device=device, requires_grad=False),

        "xy_f_train": to_tensor(xy_f, device=device, requires_grad=False),
        "src_f_train": to_tensor(src_all, device=device, requires_grad=False),

        "xy_bc_train": to_tensor(xy_bc, device=device, requires_grad=False),
        "uvTpc_bc_train": to_tensor(uvTpc_bc, device=device, requires_grad=False),

        "xy_test": to_tensor(xy_test, device=device, requires_grad=False),
        "uvTpc_test": to_tensor(uvTpc_test, device=device, requires_grad=False),
    }




class ResidualAttentionBlock(nn.Module):

    def __init__(self, hidden_dim: int, activation: str = "tanh",
                 dropout: float = 0.0, learnable_alpha: bool = True):
        super().__init__()

        if activation.lower() == "tanh":
            act = nn.Tanh
        elif activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "silu":
            act = nn.SiLU
        else:
            raise ValueError("Unsupported activation")

        self.main = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_out = self.main(x)
        gate_out = self.gate(x)
        return x + self.alpha * (main_out * gate_out)


class ResidualAttentionPINN(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        if cfg.activation.lower() == "tanh":
            act = nn.Tanh
        elif cfg.activation.lower() == "relu":
            act = nn.ReLU
        elif cfg.activation.lower() == "silu":
            act = nn.SiLU
        else:
            raise ValueError("Unsupported activation")

        self.input_layer = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            act(),
        )

        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(
                hidden_dim=cfg.hidden_dim,
                activation=cfg.activation,
                dropout=cfg.dropout,
                learnable_alpha=cfg.use_learnable_alpha
            )
            for _ in range(cfg.num_blocks)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            act(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

        self.apply(xavier_init)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        z = self.input_layer(xy)
        for blk in self.blocks:
            z = blk(z)
        return self.output_head(z)


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    return ResidualAttentionPINN(cfg).to(device)




def split_source_tensor(src: torch.Tensor):
    return src[:, 0:1], src[:, 1:2], src[:, 2:3], src[:, 3:4], src[:, 4:5]


def pde_residuals(model: nn.Module, xy: torch.Tensor, src: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:

    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    pred = model(xy)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    T = pred[:, 2:3]
    p = pred[:, 3:4]
    c = pred[:, 4:5]

    f_mass, f_u, f_v, f_T, f_c = split_source_tensor(src)

    u_x, u_y = gradients(u, xy)
    v_x, v_y = gradients(v, xy)
    T_x, T_y = gradients(T, xy)
    p_x, p_y = gradients(p, xy)
    c_x, c_y = gradients(c, xy)

    _, _, lap_u = laplacian(u, xy)
    _, _, lap_v = laplacian(v, xy)
    _, _, lap_T = laplacian(T, xy)
    _, _, lap_c = laplacian(c, xy)

    Fe_x, Fe_y = electric_forcing(u, v, T, c, xy, cfg)

    r_mass = u_x + v_y - f_mass
    r_u = u * u_x + v * u_y + p_x - cfg.nu * lap_u - Fe_x - f_u
    r_v = u * v_x + v * v_y + p_y - cfg.nu * lap_v - Fe_y - f_v
    r_T = u * T_x + v * T_y - cfg.kappa * lap_T - cfg.beta_tc * u * c - f_T
    r_c = u * c_x + v * c_y - cfg.D * lap_c - cfg.gamma_ct * T - f_c

    return {
        "mass": r_mass,
        "u_mom": r_u,
        "v_mom": r_v,
        "energy": r_T,
        "conc": r_c,
    }




def mse_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - true) ** 2)


def compute_data_loss(model: nn.Module, xy: torch.Tensor, uvTpc: torch.Tensor, batch_size: int, seed: int) -> torch.Tensor:
    total = 0.0
    nb = 0
    n = xy.shape[0]
    for idx in batch_index_iterator(n, batch_size, shuffle=True, seed=seed):
        pred = model(xy[idx])
        total = total + mse_loss(pred, uvTpc[idx])
        nb += 1
    return total / max(nb, 1)


def compute_bc_loss(model: nn.Module, xy: torch.Tensor, uvTpc: torch.Tensor, batch_size: int) -> torch.Tensor:
    total = 0.0
    nb = 0
    n = xy.shape[0]
    for idx in batch_index_iterator(n, batch_size, shuffle=False):
        pred = model(xy[idx])
        total = total + mse_loss(pred, uvTpc[idx])
        nb += 1
    return total / max(nb, 1)


def compute_pde_loss(model: nn.Module, xy: torch.Tensor, src: torch.Tensor,
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

    for idx in batch_index_iterator(n_use, cfg.pde_batch_size, shuffle=False):
        xy_batch = xy_use[idx].clone().detach().requires_grad_(True)
        src_batch = src_use[idx]

        res = pde_residuals(model, xy_batch, src_batch, cfg)

        loss_mass = torch.mean(res["mass"] ** 2)
        loss_u = torch.mean(res["u_mom"] ** 2)
        loss_v = torch.mean(res["v_mom"] ** 2)
        loss_T = torch.mean(res["energy"] ** 2)
        loss_c = torch.mean(res["conc"] ** 2)

        total_mass += loss_mass
        total_u += loss_u
        total_v += loss_v
        total_T += loss_T
        total_c += loss_c
        nb += 1

        del xy_batch, src_batch, res, loss_mass, loss_u, loss_v, loss_T, loss_c

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
    loss_data = compute_data_loss(
        model,
        dataset["xy_data_train"],
        dataset["uvTpc_data_train"],
        cfg.data_batch_size,
        cfg.seed + epoch
    )

    loss_bc = compute_bc_loss(
        model,
        dataset["xy_bc_train"],
        dataset["uvTpc_bc_train"],
        cfg.bc_batch_size
    )

    pde_dict = compute_pde_loss(
        model,
        dataset["xy_f_train"],
        dataset["src_f_train"],
        cfg,
        epoch_seed=cfg.seed + 10000 + epoch,
        n_use=cfg.n_f
    )

    loss_total = cfg.w_pde * pde_dict["loss_pde"] + cfg.w_bc * loss_bc + cfg.w_data * loss_data

    return {
        "loss_total": loss_total,
        "loss_pde": pde_dict["loss_pde"],
        "loss_bc": loss_bc,
        "loss_data": loss_data,
        "loss_mass": pde_dict["loss_mass"],
        "loss_u": pde_dict["loss_u"],
        "loss_v": pde_dict["loss_v"],
        "loss_T": pde_dict["loss_T"],
        "loss_c": pde_dict["loss_c"],
    }


def compute_val_loss(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred_val = model(dataset["xy_data_val"])
        val_data = torch.mean((pred_val - dataset["uvTpc_data_val"]) ** 2).item()

    # 验证损失以监督验证集为主；附带边界和PDE可以单独给，但这里按用户要求保留 val_loss
    total_val = cfg.w_data * val_data
    return {
        "val_loss": total_val,
        "val_data_loss": val_data,
    }




def compute_metrics_numpy(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    diff = pred - true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    l2 = float(np.linalg.norm(diff.reshape(-1), 2) / (np.linalg.norm(true.reshape(-1), 2) + 1e-14))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "L2": l2}


def predict_in_batches(model: nn.Module, xy: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        n = xy.shape[0]
        for idx in batch_index_iterator(n, batch_size, shuffle=False):
            preds.append(model(xy[idx]).detach().cpu().numpy())
    return np.vstack(preds)


def evaluate_model(model: nn.Module, xy: torch.Tensor, uvTpc_true: torch.Tensor, cfg: Config) -> Dict[str, Dict[str, float]]:
    pred = predict_in_batches(model, xy, cfg.predict_batch_size)
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


def print_metrics(title: str, metrics: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for name, vals in metrics.items():
        print(
            f"{name:>12s} | "
            f"MSE={vals['MSE']:.6e} | "
            f"RMSE={vals['RMSE']:.6e} | "
            f"MAE={vals['MAE']:.6e} | "
            f"L2={vals['L2']:.6e}"
        )
    print("=" * 100)


def save_metrics(metrics: Dict[str, Dict[str, float]], txt_path: str, csv_path: str) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("metrics\n")
        f.write("=" * 100 + "\n")
        for name, vals in metrics.items():
            f.write(f"{name}\n")
            for k, v in vals.items():
                f.write(f"  {k}: {v:.10e}\n")
            f.write("-" * 100 + "\n")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "MSE", "RMSE", "MAE", "L2"])
        for name, vals in metrics.items():
            writer.writerow([name, vals["MSE"], vals["RMSE"], vals["MAE"], vals["L2"]])




def save_txt_xyz(filepath: str, x: np.ndarray, y: np.ndarray, value: np.ndarray) -> None:
    data = np.column_stack([x.reshape(-1), y.reshape(-1), value.reshape(-1)])
    np.savetxt(filepath, data, fmt="%.8f %.8f %.8f", header="x y value", comments="")


def reshape_field(flat: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return flat.reshape(ny, nx)


def plot_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str, save_path: str, dpi: int = 300) -> None:
    plt.figure(figsize=(6.8, 5.4))
    cf = plt.contourf(X, Y, Z, levels=120, cmap="rainbow")
    plt.colorbar(cf)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_loss_curves(history: Dict[str, List[float]], fig_dir: str, dpi: int = 300) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(history["total_loss"], label="total_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Total / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_total_val.png"), dpi=dpi)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(history["pde_loss"], label="pde_loss")
    plt.plot(history["bc_loss"], label="bc_loss")
    plt.plot(history["data_loss"], label="data_loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("PDE / BC / Data Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_components.png"), dpi=dpi)
    plt.close()


def save_all_field_artifacts(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config, dirs: Dict[str, str]) -> None:
    xy_test = dataset["xy_test"]
    uvTpc_true = dataset["uvTpc_test"]

    pred = predict_in_batches(model, xy_test, cfg.predict_batch_size)
    true = uvTpc_true.detach().cpu().numpy()
    xy = xy_test.detach().cpu().numpy()

    x = xy[:, 0]
    y = xy[:, 1]

    nx = cfg.test_nx
    ny = cfg.test_ny
    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)

    names = ["u", "v", "T", "p", "c"]

    for i, name in enumerate(names):
        pred_i = pred[:, i]
        true_i = true[:, i]
        err_i = np.abs(pred_i - true_i)

        Z_pred = reshape_field(pred_i, nx, ny)
        Z_true = reshape_field(true_i, nx, ny)
        Z_err = reshape_field(err_i, nx, ny)

        plot_field(X, Y, Z_pred, f"{name}_pred", os.path.join(dirs["figures"], f"{name}_pred.png"), cfg.dpi)
        plot_field(X, Y, Z_true, f"{name}_exact", os.path.join(dirs["figures"], f"{name}_exact.png"), cfg.dpi)
        plot_field(X, Y, Z_err, f"{name}_error", os.path.join(dirs["figures"], f"{name}_error.png"), cfg.dpi)

        save_txt_xyz(os.path.join(dirs["txt"], f"{name}_pred.txt"), x, y, pred_i)
        save_txt_xyz(os.path.join(dirs["txt"], f"{name}_exact.txt"), x, y, true_i)
        save_txt_xyz(os.path.join(dirs["txt"], f"{name}_error.txt"), x, y, err_i)




def compute_supervised_mse(model: nn.Module, xy: torch.Tensor, uvTpc: torch.Tensor, batch_size: int) -> float:
    model.eval()
    total = 0.0
    nb = 0
    n = xy.shape[0]
    with torch.no_grad():
        for idx in batch_index_iterator(n, batch_size, shuffle=False):
            pred = model(xy[idx])
            total += torch.mean((pred - uvTpc[idx]) ** 2).item()
            nb += 1
    return total / max(nb, 1)


def train_model(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config,
                dirs: Dict[str, str], device: torch.device) -> Dict[str, List[float]]:
    logger = Logger(os.path.join(dirs["logs"], "training_log.txt"))

    optimizer = optim.Adam(model.parameters(), lr=cfg.adam_lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
        )

    best_val = float("inf")
    best_epoch = -1

    best_model_path = os.path.join(dirs["models"], "best_model.pth")
    final_model_path = os.path.join(dirs["models"], "final_model.pth")

    history = {
        "epoch": [],
        "total_loss": [],
        "pde_loss": [],
        "bc_loss": [],
        "data_loss": [],
        "mass_loss": [],
        "u_loss": [],
        "v_loss": [],
        "T_loss": [],
        "c_loss": [],
        "val_loss": [],
        "val_data_loss": [],
        "train_mse": [],
        "val_mse": [],
        "lr": [],
    }

    with open(os.path.join(dirs["logs"], "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger.log(f"Device: {device}")
    logger.log(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.log("Model: Residual Attention PINN")
    logger.log("-" * 120)

    total_train_start = time.time()
    adam_start = time.time()


    for epoch in range(1, cfg.adam_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        losses = compute_train_losses(model, dataset, cfg, epoch)
        losses["loss_total"].backward()

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        val_info = compute_val_loss(model, dataset, cfg)
        train_mse = compute_supervised_mse(model, dataset["xy_data_train"], dataset["uvTpc_data_train"], cfg.data_batch_size)
        val_mse = compute_supervised_mse(model, dataset["xy_data_val"], dataset["uvTpc_data_val"], cfg.data_batch_size)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["total_loss"].append(losses["loss_total"].item())
        history["pde_loss"].append(losses["loss_pde"].item())
        history["bc_loss"].append(losses["loss_bc"].item())
        history["data_loss"].append(losses["loss_data"].item())
        history["mass_loss"].append(losses["loss_mass"].item())
        history["u_loss"].append(losses["loss_u"].item())
        history["v_loss"].append(losses["loss_v"].item())
        history["T_loss"].append(losses["loss_T"].item())
        history["c_loss"].append(losses["loss_c"].item())
        history["val_loss"].append(val_info["val_loss"])
        history["val_data_loss"].append(val_info["val_data_loss"])
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["lr"].append(current_lr)

        if val_info["val_loss"] < best_val:
            best_val = val_info["val_loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        if epoch == 1 or epoch % cfg.print_every == 0 or epoch == cfg.adam_epochs:
            logger.log(
                f"Epoch [{epoch:5d}/{cfg.adam_epochs}] | "
                f"Total={losses['loss_total'].item():.6e} | "
                f"PDE={losses['loss_pde'].item():.6e} | "
                f"BC={losses['loss_bc'].item():.6e} | "
                f"Data={losses['loss_data'].item():.6e} | "
                f"Mass={losses['loss_mass'].item():.6e} | "
                f"Ux={losses['loss_u'].item():.6e} | "
                f"Vy={losses['loss_v'].item():.6e} | "
                f"T={losses['loss_T'].item():.6e} | "
                f"C={losses['loss_c'].item():.6e} | "
                f"Val={val_info['val_loss']:.6e} | "
                f"TrainMSE={train_mse:.6e} | "
                f"ValMSE={val_mse:.6e} | "
                f"LR={current_lr:.3e}"
            )

        if device.type == "cuda" and epoch % cfg.empty_cache_every == 0:
            torch.cuda.empty_cache()

    adam_time = time.time() - adam_start


    lbfgs_time = 0.0
    if cfg.use_lbfgs:
        logger.log("Starting LBFGS refinement...")
        lbfgs_start = time.time()

        lbfgs = optim.LBFGS(
            model.parameters(),
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        def closure():
            lbfgs.zero_grad()
            losses = compute_train_losses(model, dataset, cfg, epoch=0)
            losses["loss_total"].backward()
            return losses["loss_total"]

        lbfgs.step(closure)
        lbfgs_time = time.time() - lbfgs_start

        val_info = compute_val_loss(model, dataset, cfg)
        if val_info["val_loss"] < best_val:
            best_val = val_info["val_loss"]
            best_epoch = cfg.adam_epochs
            torch.save(model.state_dict(), best_model_path)

    total_train_time = time.time() - total_train_start
    torch.save(model.state_dict(), final_model_path)


    train_log_csv = os.path.join(dirs["logs"], "train_log.csv")
    with open(train_log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "total_loss", "pde_loss", "bc_loss", "data_loss",
            "mass_loss", "u_loss", "v_loss", "T_loss", "c_loss",
            "val_loss", "val_data_loss", "train_mse", "val_mse", "lr"
        ])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["total_loss"][i],
                history["pde_loss"][i],
                history["bc_loss"][i],
                history["data_loss"][i],
                history["mass_loss"][i],
                history["u_loss"][i],
                history["v_loss"][i],
                history["T_loss"][i],
                history["c_loss"][i],
                history["val_loss"][i],
                history["val_data_loss"][i],
                history["train_mse"][i],
                history["val_mse"][i],
                history["lr"][i],
            ])


    with open(os.path.join(dirs["logs"], "time_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"adam_time_sec: {adam_time:.6f}\n")
        f.write(f"lbfgs_time_sec: {lbfgs_time:.6f}\n")
        f.write(f"total_train_time_sec: {total_train_time:.6f}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_loss: {best_val:.10e}\n")

    logger.log("=" * 120)
    logger.log(f"Best epoch: {best_epoch}")
    logger.log(f"Best validation loss: {best_val:.10e}")
    logger.log(f"Adam time (s): {adam_time:.6f}")
    logger.log(f"LBFGS time (s): {lbfgs_time:.6f}")
    logger.log(f"Total train time (s): {total_train_time:.6f}")
    logger.log("=" * 120)

    history["adam_time_sec"] = [adam_time]
    history["lbfgs_time_sec"] = [lbfgs_time]
    history["total_train_time_sec"] = [total_train_time]
    history["best_epoch"] = [best_epoch]
    history["best_val_loss"] = [best_val]

    return history



def main():
    program_start = time.time()

    set_seed(CFG.seed)
    device = get_device(CFG.use_cuda)
    dirs = make_dirs(CFG.results_dir)

    dataset = generate_dataset(CFG, device)
    model = build_model(CFG, device)

    history = train_model(model, dataset, CFG, dirs, device)


    best_model_path = os.path.join(dirs["models"], "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))


    test_start = time.time()
    test_metrics = evaluate_model(model, dataset["xy_test"], dataset["uvTpc_test"], CFG)
    test_time = time.time() - test_start

    print_metrics("Test Metrics", test_metrics)

    save_metrics(
        test_metrics,
        txt_path=os.path.join(dirs["txt"], "metrics.txt"),
        csv_path=os.path.join(dirs["logs"], "metrics.csv"),
    )


    save_all_field_artifacts(model, dataset, CFG, dirs)
    plot_loss_curves(history, dirs["figures"], CFG.dpi)

    total_runtime = time.time() - program_start


    with open(os.path.join(dirs["logs"], "time_log.txt"), "a", encoding="utf-8") as f:
        f.write(f"test_time_sec: {test_time:.6f}\n")
        f.write(f"total_runtime_sec: {total_runtime:.6f}\n")

    print("\nOutputs saved to:", dirs["base"])
    print(f"Test time (s): {test_time:.6f}")
    print(f"Total runtime (s): {total_runtime:.6f}")


if __name__ == "__main__":
    main()