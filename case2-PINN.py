

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import math
import time
import json
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


    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0


    nx: int = 101
    ny: int = 101


    train_ratio: float = 0.7


    n_data_samples: int = 8000
    n_collocation: int = 12000
    n_boundary_each_side: int = 500


    data_batch_size: int = 1024
    bc_batch_size: int = 1024
    pde_batch_size: int = 64
    pde_points_per_epoch: int = 1024


    exact_chunk_size: int = 2048
    source_chunk_size: int = 256
    predict_chunk_size: int = 4096


    A_psi: float = 1.0
    A_T: float = 0.8
    A_c: float = 0.7
    A_p: float = 0.6
    A_e: float = 1.0


    Re: float = 20.0
    Pr: float = 3.0
    Sc: float = 2.0
    M: float = 1.5
    Ri_T: float = 0.8
    Gamma_Tc: float = 0.2
    lambda_c: float = 0.1
    chi: float = 0.4


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


    scheduler_step_size: int = 1000
    scheduler_gamma: float = 0.7


    lambda_pde: float = 1.0
    lambda_data: float = 10.0
    lambda_bc: float = 5.0


    results_dir: str = "results"
    dpi: int = 300
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




def envelope_B(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    return (x ** 2) * ((1.0 - x) ** 2) * (y ** 2) * ((1.0 - y) ** 2)


def psi_exact(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:

    pi = math.pi
    B = envelope_B(x, y)
    inside = (
        torch.sin(pi * (x + 0.23 * y))
        + 0.37 * torch.cos(2.0 * pi * y + 0.19 * pi * x)
        + 0.12 * torch.exp(0.18 * x - 0.11 * y) * torch.sin(pi * x * y)
    )
    return cfg.A_psi * B * inside


def T_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:

    pi = math.pi
    B = envelope_B(x, y)
    inside = (
        torch.cos(pi * (1.11 * x + 0.93 * y))
        + 0.21 * torch.sin(2.0 * pi * x * y)
        + 0.09 * torch.exp(0.14 * x - 0.08 * y) * torch.cos(pi * y)
    )
    return cfg.A_T * B * inside


def c_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:

    pi = math.pi
    B = envelope_B(x, y)
    inside = (
        torch.sin(pi * (0.87 * x + 1.16 * y))
        + 0.18 * torch.cos(2.0 * pi * x * y + 0.17 * pi * x)
        + 0.07 * torch.exp(-0.09 * x + 0.13 * y) * torch.sin(pi * (x + y))
    )
    return cfg.A_c * B * inside


def p_star_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:

    pi = math.pi
    xs = x - 0.5
    ys = y - 0.5
    inside = (
        xs * ys
        + 0.15 * xs ** 3
        - 0.11 * ys ** 3
        + 0.08 * torch.sin(pi * x + 0.27 * pi * y)
        + 0.05 * torch.exp(0.09 * x - 0.04 * y) * torch.cos(pi * x * y)
    )
    return cfg.A_p * inside


def phi_e_exact(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:

    pi = math.pi
    inside = (
        torch.sin(3.0 * pi * x + pi / 7.0) * torch.sin(pi * y)
        + 0.35 * torch.cos(pi * x + 0.17 * pi * y) * torch.sin(2.0 * pi * y)
        + 0.12 * torch.exp(-0.2 * (1.0 - x)) * torch.sin(2.0 * pi * (x * y + y))
    )
    return cfg.A_e * inside


def compute_pressure_mean_on_grid(cfg: Config, device: torch.device) -> torch.Tensor:

    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, device=device)
    y = torch.linspace(cfg.y_min, cfg.y_max, cfg.ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    p_star = p_star_formula(X.reshape(-1, 1), Y.reshape(-1, 1), cfg)
    return torch.mean(p_star).detach()


def exact_solution(xy: torch.Tensor, p_mean: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:

    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    x = xy[:, 0:1]
    y = xy[:, 1:2]

    psi = psi_exact(x, y, cfg)
    grad_psi = safe_grad(psi, xy, create_graph=True, retain_graph=True)
    psi_x = grad_psi[:, 0:1]
    psi_y = grad_psi[:, 1:2]

    u = psi_y
    v = -psi_x
    T = T_exact_formula(x, y, cfg)
    c = c_exact_formula(x, y, cfg)
    p_star = p_star_formula(x, y, cfg)
    p = p_star - p_mean

    phi_e = phi_e_exact(x, y, cfg)
    grad_phi = safe_grad(phi_e, xy, create_graph=True, retain_graph=True)
    Ex = -grad_phi[:, 0:1]
    Ey = -grad_phi[:, 1:2]

    return {
        "u": u,
        "v": v,
        "T": T,
        "p": p,
        "c": c,
        "phi_e": phi_e,
        "Ex": Ex,
        "Ey": Ey,
    }




def compute_source_terms(xy: torch.Tensor, p_mean: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:

    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    sol = exact_solution(xy, p_mean, cfg)
    u, v, T, p, c = sol["u"], sol["v"], sol["T"], sol["p"], sol["c"]
    Ex, Ey = sol["Ex"], sol["Ey"]

    u_x, u_y = gradients(u, xy)
    v_x, v_y = gradients(v, xy)
    T_x, T_y = gradients(T, xy)
    p_x, p_y = gradients(p, xy)
    c_x, c_y = gradients(c, xy)

    _, _, lap_u = laplacian(u, xy)
    _, _, lap_v = laplacian(v, xy)
    _, _, lap_T = laplacian(T, xy)
    _, _, lap_c = laplacian(c, xy)

    cEx = c * Ex
    cEy = c * Ey
    div_cE = safe_grad(cEx, xy, create_graph=True, retain_graph=True)[:, 0:1] + \
             safe_grad(cEy, xy, create_graph=True, retain_graph=True)[:, 1:2]

    Su = u * u_x + v * u_y + p_x - (1.0 / cfg.Re) * lap_u - cfg.M * c * Ex
    Sv = u * v_x + v * v_y + p_y - (1.0 / cfg.Re) * lap_v - cfg.M * c * Ey - cfg.Ri_T * T
    ST = u * T_x + v * T_y - (1.0 / (cfg.Re * cfg.Pr)) * lap_T - cfg.Gamma_Tc * c
    Sc = u * c_x + v * c_y - (1.0 / (cfg.Re * cfg.Sc)) * lap_c + cfg.lambda_c * c - cfg.chi * div_cE

    continuity_exact = u_x + v_y

    return {
        "Su": Su,
        "Sv": Sv,
        "ST": ST,
        "Sc": Sc,
        "continuity_exact": continuity_exact
    }




def lhs_sampling(n: int, dim: int = 2, low: float = 0.0, high: float = 1.0, seed: int = 42) -> np.ndarray:

    rng = np.random.default_rng(seed)
    result = np.zeros((n, dim), dtype=np.float32)
    for j in range(dim):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.random(n)) / n
    result = low + (high - low) * result
    return result.astype(np.float32)


def generate_boundary_points(cfg: Config, seed: int = 42) -> np.ndarray:

    rng = np.random.default_rng(seed)
    n = cfg.n_boundary_each_side

    xb1 = rng.uniform(cfg.x_min, cfg.x_max, size=(n, 1)).astype(np.float32)
    yb1 = np.full((n, 1), cfg.y_min, dtype=np.float32)

    xb2 = rng.uniform(cfg.x_min, cfg.x_max, size=(n, 1)).astype(np.float32)
    yb2 = np.full((n, 1), cfg.y_max, dtype=np.float32)

    yb3 = rng.uniform(cfg.y_min, cfg.y_max, size=(n, 1)).astype(np.float32)
    xb3 = np.full((n, 1), cfg.x_min, dtype=np.float32)

    yb4 = rng.uniform(cfg.y_min, cfg.y_max, size=(n, 1)).astype(np.float32)
    xb4 = np.full((n, 1), cfg.x_max, dtype=np.float32)

    return np.vstack([
        np.hstack([xb1, yb1]),
        np.hstack([xb2, yb2]),
        np.hstack([xb3, yb3]),
        np.hstack([xb4, yb4]),
    ]).astype(np.float32)


def exact_solution_on_points_numpy(xy_np: np.ndarray, p_mean: torch.Tensor,
                                   cfg: Config, device: torch.device,
                                   chunk_size: int = 2048) -> np.ndarray:

    out = []
    for start in range(0, xy_np.shape[0], chunk_size):
        end = min(start + chunk_size, xy_np.shape[0])
        xy_chunk = to_tensor(xy_np[start:end], device=device, requires_grad=True)
        sol = exact_solution(xy_chunk, p_mean, cfg)
        vals = torch.cat([sol["u"], sol["v"], sol["T"], sol["p"], sol["c"]], dim=1).detach()
        out.append(vals.cpu().numpy())
        del xy_chunk, sol, vals
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.vstack(out)


def source_terms_on_points_numpy(xy_np: np.ndarray, p_mean: torch.Tensor,
                                 cfg: Config, device: torch.device,
                                 chunk_size: int = 256) -> Dict[str, np.ndarray]:

    Su_all, Sv_all, ST_all, Sc_all = [], [], [], []
    for start in range(0, xy_np.shape[0], chunk_size):
        end = min(start + chunk_size, xy_np.shape[0])
        xy_chunk = to_tensor(xy_np[start:end], device=device, requires_grad=True)
        src = compute_source_terms(xy_chunk, p_mean, cfg)
        Su_all.append(src["Su"].detach().cpu().numpy())
        Sv_all.append(src["Sv"].detach().cpu().numpy())
        ST_all.append(src["ST"].detach().cpu().numpy())
        Sc_all.append(src["Sc"].detach().cpu().numpy())
        del xy_chunk, src
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "Su": np.vstack(Su_all),
        "Sv": np.vstack(Sv_all),
        "ST": np.vstack(ST_all),
        "Sc": np.vstack(Sc_all),
    }


def generate_dataset(cfg: Config, device: torch.device, p_mean: torch.Tensor) -> Dict[str, torch.Tensor]:


    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=np.float32)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    xy_grid = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)
    uvTpc_grid_np = exact_solution_on_points_numpy(
        xy_grid, p_mean, cfg, device=device, chunk_size=cfg.exact_chunk_size
    )


    data_xy = lhs_sampling(cfg.n_data_samples, dim=2, low=cfg.x_min, high=cfg.x_max, seed=cfg.seed + 10)
    data_uvTpc = exact_solution_on_points_numpy(
        data_xy, p_mean, cfg, device=device, chunk_size=cfg.exact_chunk_size
    )


    idx = np.arange(cfg.n_data_samples)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(idx)
    n_train = int(cfg.train_ratio * cfg.n_data_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    xy_train = data_xy[train_idx]
    uvTpc_train = data_uvTpc[train_idx]

    xy_val = data_xy[val_idx]
    uvTpc_val = data_uvTpc[val_idx]


    xy_collocation = lhs_sampling(cfg.n_collocation, dim=2, low=cfg.x_min, high=cfg.x_max, seed=cfg.seed + 20)
    source_np = source_terms_on_points_numpy(
        xy_collocation, p_mean, cfg, device=device, chunk_size=cfg.source_chunk_size
    )


    xy_boundary = generate_boundary_points(cfg, seed=cfg.seed + 30)
    uvTpc_boundary = exact_solution_on_points_numpy(
        xy_boundary, p_mean, cfg, device=device, chunk_size=cfg.exact_chunk_size
    )

    return {
        "X": X,
        "Y": Y,

        "xy_grid": to_tensor(xy_grid, device=device, requires_grad=False),
        "uvTpc_grid": to_tensor(uvTpc_grid_np, device=device, requires_grad=False),

        "xy_train": to_tensor(xy_train, device=device, requires_grad=False),
        "uvTpc_train": to_tensor(uvTpc_train, device=device, requires_grad=False),

        "xy_val": to_tensor(xy_val, device=device, requires_grad=False),
        "uvTpc_val": to_tensor(uvTpc_val, device=device, requires_grad=False),

        "xy_collocation": to_tensor(xy_collocation, device=device, requires_grad=False),
        "Su_collocation": to_tensor(source_np["Su"], device=device, requires_grad=False),
        "Sv_collocation": to_tensor(source_np["Sv"], device=device, requires_grad=False),
        "ST_collocation": to_tensor(source_np["ST"], device=device, requires_grad=False),
        "Sc_collocation": to_tensor(source_np["Sc"], device=device, requires_grad=False),

        "xy_boundary": to_tensor(xy_boundary, device=device, requires_grad=False),
        "uvTpc_boundary": to_tensor(uvTpc_boundary, device=device, requires_grad=False),
    }




class PINN(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        if cfg.activation.lower() == "tanh":
            act_cls = nn.Tanh
        elif cfg.activation.lower() == "relu":
            act_cls = nn.ReLU
        else:
            raise ValueError("Only tanh or relu activation is supported.")

        layers = []
        in_dim = cfg.input_dim


        layers.append(nn.Linear(in_dim, cfg.hidden_dim))
        layers.append(act_cls())
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        in_dim = cfg.hidden_dim

        for _ in range(cfg.num_hidden_layers - 1):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(act_cls())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.hidden_dim


        layers.append(nn.Linear(in_dim, cfg.output_dim))

        self.net = nn.Sequential(*layers)
        self.apply(xavier_init)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    return PINN(cfg).to(device)




def pde_residuals(
    model: nn.Module,
    xy: torch.Tensor,
    Su: torch.Tensor,
    Sv: torch.Tensor,
    ST: torch.Tensor,
    Sc: torch.Tensor,
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

    x = xy[:, 0:1]
    y = xy[:, 1:2]
    phi_e = phi_e_exact(x, y, cfg)
    grad_phi = safe_grad(phi_e, xy, create_graph=True, retain_graph=True)
    Ex = -grad_phi[:, 0:1]
    Ey = -grad_phi[:, 1:2]

    cEx = c * Ex
    cEy = c * Ey
    div_cE = safe_grad(cEx, xy, create_graph=True, retain_graph=True)[:, 0:1] + \
             safe_grad(cEy, xy, create_graph=True, retain_graph=True)[:, 1:2]

    r_cont = u_x + v_y
    r_u = u * u_x + v * u_y + p_x - (1.0 / cfg.Re) * lap_u - cfg.M * c * Ex - Su
    r_v = u * v_x + v * v_y + p_y - (1.0 / cfg.Re) * lap_v - cfg.M * c * Ey - cfg.Ri_T * T - Sv
    r_T = u * T_x + v * T_y - (1.0 / (cfg.Re * cfg.Pr)) * lap_T - cfg.Gamma_Tc * c - ST
    r_c = u * c_x + v * c_y - (1.0 / (cfg.Re * cfg.Sc)) * lap_c + cfg.lambda_c * c - cfg.chi * div_cE - Sc

    return {
        "continuity": r_cont,
        "momentum_u": r_u,
        "momentum_v": r_v,
        "temperature": r_T,
        "concentration": r_c
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


def compute_pde_loss_batched(model: nn.Module, dataset: Dict[str, torch.Tensor],
                             cfg: Config, epoch_seed: int) -> Dict[str, torch.Tensor]:

    n_all = dataset["xy_collocation"].shape[0]
    rng = np.random.default_rng(epoch_seed)
    n_use = min(cfg.pde_points_per_epoch, n_all)
    chosen = rng.choice(n_all, size=n_use, replace=False)

    xy_use = dataset["xy_collocation"][chosen]
    Su_use = dataset["Su_collocation"][chosen]
    Sv_use = dataset["Sv_collocation"][chosen]
    ST_use = dataset["ST_collocation"][chosen]
    Sc_use = dataset["Sc_collocation"][chosen]

    total_cont = 0.0
    total_u = 0.0
    total_v = 0.0
    total_T = 0.0
    total_c = 0.0
    nb = 0

    for idx in batch_iterator(n_use, cfg.pde_batch_size, shuffle=False):
        xy_batch = xy_use[idx].clone().detach().requires_grad_(True)
        Su_batch = Su_use[idx]
        Sv_batch = Sv_use[idx]
        ST_batch = ST_use[idx]
        Sc_batch = Sc_use[idx]

        res = pde_residuals(model, xy_batch, Su_batch, Sv_batch, ST_batch, Sc_batch, cfg)

        loss_cont = torch.mean(res["continuity"] ** 2)
        loss_u = torch.mean(res["momentum_u"] ** 2)
        loss_v = torch.mean(res["momentum_v"] ** 2)
        loss_T = torch.mean(res["temperature"] ** 2)
        loss_c = torch.mean(res["concentration"] ** 2)

        total_cont = total_cont + loss_cont
        total_u = total_u + loss_u
        total_v = total_v + loss_v
        total_T = total_T + loss_T
        total_c = total_c + loss_c
        nb += 1

        del xy_batch, Su_batch, Sv_batch, ST_batch, Sc_batch, res, loss_cont, loss_u, loss_v, loss_T, loss_c

    loss_cont = total_cont / max(nb, 1)
    loss_u = total_u / max(nb, 1)
    loss_v = total_v / max(nb, 1)
    loss_T = total_T / max(nb, 1)
    loss_c = total_c / max(nb, 1)
    loss_pde = loss_cont + loss_u + loss_v + loss_T + loss_c

    return {
        "loss_pde": loss_pde,
        "loss_cont": loss_cont,
        "loss_u": loss_u,
        "loss_v": loss_v,
        "loss_T": loss_T,
        "loss_c": loss_c,
    }


def compute_losses(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config, epoch: int) -> Dict[str, torch.Tensor]:

    loss_data = compute_data_loss_batched(
        model,
        dataset["xy_train"],
        dataset["uvTpc_train"],
        batch_size=cfg.data_batch_size,
        epoch_seed=cfg.seed + epoch
    )

    pde_losses = compute_pde_loss_batched(
        model,
        dataset,
        cfg,
        epoch_seed=cfg.seed + 10000 + epoch
    )

    loss_bc = compute_bc_loss_batched(
        model,
        dataset["xy_boundary"],
        dataset["uvTpc_boundary"],
        batch_size=cfg.bc_batch_size
    )

    loss_total = (
        cfg.lambda_pde * pde_losses["loss_pde"]
        + cfg.lambda_data * loss_data
        + cfg.lambda_bc * loss_bc
    )

    return {
        "loss_total": loss_total,
        "loss_pde": pde_losses["loss_pde"],
        "loss_data": loss_data,
        "loss_bc": loss_bc,
        "loss_cont": pde_losses["loss_cont"],
        "loss_u": pde_losses["loss_u"],
        "loss_v": pde_losses["loss_v"],
        "loss_T": pde_losses["loss_T"],
        "loss_c": pde_losses["loss_c"],
    }



def compute_metrics_numpy(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:

    diff = pred - true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    denom = np.linalg.norm(true.reshape(-1), ord=2)
    l2 = float(np.linalg.norm(diff.reshape(-1), ord=2) / (denom + 1e-14))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "L2": l2}


def predict_in_batches(model: nn.Module, xy: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
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



def save_txt_data(x: np.ndarray, y: np.ndarray, value: np.ndarray, filepath: str) -> None:

    data = np.column_stack([x.reshape(-1), y.reshape(-1), value.reshape(-1)])
    np.savetxt(filepath, data, fmt="%.8f %.8f %.8f")




def plot_single_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                      title: str, save_path: str, dpi: int = 300) -> None:
    plt.figure(figsize=(7, 6))
    contour = plt.contourf(X, Y, Z, levels=120, cmap="rainbow")
    plt.colorbar(contour)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], fig_dir: str, dpi: int = 300) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(history["loss_total"], label="Total")
    plt.plot(history["loss_pde"], label="PDE")
    plt.plot(history["loss_data"], label="Data")
    plt.plot(history["loss_bc"], label="BC")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_curves.png"), dpi=dpi)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(history["loss_cont"], label="Continuity")
    plt.plot(history["loss_u"], label="Momentum-x")
    plt.plot(history["loss_v"], label="Momentum-y")
    plt.plot(history["loss_T"], label="Temperature")
    plt.plot(history["loss_c"], label="Concentration")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PDE Component Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pde_component_loss_curves.png"), dpi=dpi)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(history["train_mse"], label="Train MSE")
    plt.plot(history["val_mse"], label="Val MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Train / Validation MSE Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "train_val_mse_curves.png"), dpi=dpi)
    plt.close()


def plot_and_save_field_triplet(
    X: np.ndarray, Y: np.ndarray,
    pred_field: np.ndarray, true_field: np.ndarray,
    name: str, fig_dir: str, txt_dir: str, dpi: int
) -> None:
    err_field = np.abs(pred_field - true_field)

    plot_single_field(X, Y, pred_field, f"{name} Prediction", os.path.join(fig_dir, f"{name}_pred.png"), dpi=dpi)
    plot_single_field(X, Y, true_field, f"{name} True", os.path.join(fig_dir, f"{name}_true.png"), dpi=dpi)
    plot_single_field(X, Y, err_field, f"{name} Absolute Error", os.path.join(fig_dir, f"{name}_error.png"), dpi=dpi)

    save_txt_data(X, Y, pred_field, os.path.join(txt_dir, f"{name}_pred.txt"))
    save_txt_data(X, Y, true_field, os.path.join(txt_dir, f"{name}_true.txt"))
    save_txt_data(X, Y, err_field, os.path.join(txt_dir, f"{name}_error.txt"))


def pde_residual_fields_on_grid(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config) -> Dict[str, np.ndarray]:

    X, Y = dataset["X"], dataset["Y"]
    xy_np = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)
    device = dataset["xy_grid"].device
    p_mean = compute_pressure_mean_on_grid(cfg, device)

    r_cont_all, r_u_all, r_v_all, r_T_all, r_c_all = [], [], [], [], []
    n = xy_np.shape[0]
    chunk = 1024
    model.eval()

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        xy_chunk = to_tensor(xy_np[start:end], device=device, requires_grad=True)
        src = compute_source_terms(xy_chunk, p_mean, cfg)
        res = pde_residuals(
            model, xy_chunk,
            src["Su"], src["Sv"], src["ST"], src["Sc"],
            cfg
        )
        r_cont_all.append(res["continuity"].detach().cpu().numpy())
        r_u_all.append(res["momentum_u"].detach().cpu().numpy())
        r_v_all.append(res["momentum_v"].detach().cpu().numpy())
        r_T_all.append(res["temperature"].detach().cpu().numpy())
        r_c_all.append(res["concentration"].detach().cpu().numpy())
        del xy_chunk, src, res
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ny, nx = cfg.ny, cfg.nx
    return {
        "r_continuity": np.vstack(r_cont_all).reshape(ny, nx),
        "r_momentum_u": np.vstack(r_u_all).reshape(ny, nx),
        "r_momentum_v": np.vstack(r_v_all).reshape(ny, nx),
        "r_temperature": np.vstack(r_T_all).reshape(ny, nx),
        "r_concentration": np.vstack(r_c_all).reshape(ny, nx),
    }


def plot_results(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config, dirs: Dict[str, str]) -> None:

    pred = predict_in_batches(model, dataset["xy_grid"], batch_size=cfg.predict_chunk_size)
    true = dataset["uvTpc_grid"].detach().cpu().numpy()

    X = dataset["X"]
    Y = dataset["Y"]
    nx, ny = cfg.nx, cfg.ny
    names = ["u", "v", "T", "p", "c"]

    for i, name in enumerate(names):
        pred_field = pred[:, i].reshape(ny, nx)
        true_field = true[:, i].reshape(ny, nx)
        plot_and_save_field_triplet(X, Y, pred_field, true_field, name, dirs["figures"], dirs["txt"], cfg.dpi)

    # Combined prediction figure
    plt.figure(figsize=(18, 10))
    for i, name in enumerate(names):
        field = pred[:, i].reshape(ny, nx)
        plt.subplot(2, 3, i + 1)
        contour = plt.contourf(X, Y, field, levels=120, cmap="rainbow")
        plt.colorbar(contour)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{name} Prediction")
    plt.subplot(2, 3, 6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["figures"], "combined_prediction.png"), dpi=cfg.dpi)
    plt.close()

    # PDE residual fields
    residual_fields = pde_residual_fields_on_grid(model, dataset, cfg)
    for rname, rfield in residual_fields.items():
        plot_single_field(X, Y, rfield, f"{rname} Field", os.path.join(dirs["figures"], f"{rname}.png"), dpi=cfg.dpi)
        save_txt_data(X, Y, rfield, os.path.join(dirs["txt"], f"{rname}.txt"))




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

    best_val_mse = float("inf")
    best_epoch = -1
    best_model_path = os.path.join(dirs["models"], "best_model.pt")
    final_model_path = os.path.join(dirs["models"], "final_model.pt")

    history = {
        "loss_total": [],
        "loss_pde": [],
        "loss_data": [],
        "loss_bc": [],
        "loss_cont": [],
        "loss_u": [],
        "loss_v": [],
        "loss_T": [],
        "loss_c": [],
        "train_mse": [],
        "val_mse": [],
        "epoch_time": [],
    }


    with open(os.path.join(dirs["logs"], "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger.log(f"Device: {device}")
    logger.log(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.log(f"Grid size: {cfg.nx} x {cfg.ny}")
    logger.log(f"Supervised data points: {cfg.n_data_samples}")
    logger.log(f"Collocation points: {cfg.n_collocation}")
    logger.log(f"PDE points per epoch: {cfg.pde_points_per_epoch}")
    logger.log(f"Boundary points: {4 * cfg.n_boundary_each_side}")
    logger.log(f"Train ratio: {cfg.train_ratio}")
    logger.log(f"Epochs: {cfg.epochs}")
    logger.log(f"Learning rate: {cfg.lr}")
    logger.log("Network type: Pure MLP-PINN")
    logger.log("-" * 120)

    train_start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        losses = compute_losses(model, dataset, cfg, epoch)
        losses["loss_total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        train_mse = compute_supervised_mse_batched(
            model, dataset["xy_train"], dataset["uvTpc_train"], batch_size=cfg.data_batch_size
        )
        val_mse = compute_supervised_mse_batched(
            model, dataset["xy_val"], dataset["uvTpc_val"], batch_size=cfg.data_batch_size
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        epoch_time = time.time() - epoch_start

        history["loss_total"].append(losses["loss_total"].item())
        history["loss_pde"].append(losses["loss_pde"].item())
        history["loss_data"].append(losses["loss_data"].item())
        history["loss_bc"].append(losses["loss_bc"].item())
        history["loss_cont"].append(losses["loss_cont"].item())
        history["loss_u"].append(losses["loss_u"].item())
        history["loss_v"].append(losses["loss_v"].item())
        history["loss_T"].append(losses["loss_T"].item())
        history["loss_c"].append(losses["loss_c"].item())
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["epoch_time"].append(epoch_time)

        if epoch == 1 or epoch % cfg.print_every == 0 or epoch == cfg.epochs:
            logger.log(
                f"Epoch [{epoch:5d}/{cfg.epochs}] | "
                f"Total={losses['loss_total'].item():.6e} | "
                f"PDE={losses['loss_pde'].item():.6e} | "
                f"Data={losses['loss_data'].item():.6e} | "
                f"BC={losses['loss_bc'].item():.6e} | "
                f"Cont={losses['loss_cont'].item():.6e} | "
                f"Ux={losses['loss_u'].item():.6e} | "
                f"Vy={losses['loss_v'].item():.6e} | "
                f"T={losses['loss_T'].item():.6e} | "
                f"C={losses['loss_c'].item():.6e} | "
                f"TrainMSE={train_mse:.6e} | "
                f"ValMSE={val_mse:.6e} | "
                f"EpochTime={epoch_time:.3f}s"
            )

        if device.type == "cuda" and (epoch % cfg.empty_cache_every == 0):
            torch.cuda.empty_cache()

    train_end_time = time.time()
    total_training_time = train_end_time - train_start_time

    torch.save(model.state_dict(), final_model_path)

    with open(os.path.join(dirs["logs"], "training_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Training Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Training start timestamp: {train_start_time:.6f}\n")
        f.write(f"Training end timestamp: {train_end_time:.6f}\n")
        f.write(f"Total training time (s): {total_training_time:.6f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation MSE: {best_val_mse:.10e}\n")
        f.write(f"Average epoch time (s): {np.mean(history['epoch_time']):.6f}\n")

    # Save loss history
    history_csv = os.path.join(dirs["logs"], "loss_history.csv")
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "loss_total", "loss_pde", "loss_data", "loss_bc",
            "loss_cont", "loss_u", "loss_v", "loss_T", "loss_c",
            "train_mse", "val_mse", "epoch_time"
        ])
        for i in range(cfg.epochs):
            writer.writerow([
                i + 1,
                history["loss_total"][i],
                history["loss_pde"][i],
                history["loss_data"][i],
                history["loss_bc"][i],
                history["loss_cont"][i],
                history["loss_u"][i],
                history["loss_v"][i],
                history["loss_T"][i],
                history["loss_c"][i],
                history["train_mse"][i],
                history["val_mse"][i],
                history["epoch_time"][i],
            ])

    logger.log("=" * 120)
    logger.log(f"Best epoch: {best_epoch}")
    logger.log(f"Best validation MSE: {best_val_mse:.10e}")
    logger.log(f"Total training time (s): {total_training_time:.6f}")
    logger.log(f"Best model saved to: {best_model_path}")
    logger.log(f"Final model saved to: {final_model_path}")
    logger.log("=" * 120)

    history["best_epoch"] = [best_epoch]
    history["best_val_mse"] = [best_val_mse]
    history["training_time_sec"] = [total_training_time]

    return history




def main():
    total_start = time.time()

    set_seed(CFG.seed)
    device = get_device(CFG.use_cuda)
    dirs = make_dirs(CFG.results_dir)


    p_mean = compute_pressure_mean_on_grid(CFG, device)


    dataset = generate_dataset(CFG, device, p_mean)


    model = build_model(CFG, device)


    history = train_model(model, dataset, CFG, dirs, device)


    best_model_path = os.path.join(dirs["models"], "best_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))


    train_metrics = evaluate_on_points(model, dataset["xy_train"], dataset["uvTpc_train"], CFG)
    val_metrics = evaluate_on_points(model, dataset["xy_val"], dataset["uvTpc_val"], CFG)
    grid_metrics = evaluate_on_points(model, dataset["xy_grid"], dataset["uvTpc_grid"], CFG)

    print_metrics("Training-set Metrics", train_metrics)
    print_metrics("Validation-set Metrics", val_metrics)
    print_metrics("Full-grid Metrics", grid_metrics)

    save_metrics(train_metrics, os.path.join(dirs["logs"], "metrics_train"))
    save_metrics(val_metrics, os.path.join(dirs["logs"], "metrics_val"))
    save_metrics(grid_metrics, os.path.join(dirs["logs"], "metrics_grid"))


    plot_training_curves(history, dirs["figures"], dpi=CFG.dpi)
    plot_results(model, dataset, CFG, dirs)


    total_end = time.time()
    total_runtime = total_end - total_start

    with open(os.path.join(dirs["logs"], "runtime.txt"), "w", encoding="utf-8") as f:
        f.write("Runtime Information\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total program start timestamp: {total_start:.6f}\n")
        f.write(f"Total program end timestamp: {total_end:.6f}\n")
        f.write(f"Total runtime (s): {total_runtime:.6f}\n")
        f.write(f"Training time (s): {history['training_time_sec'][0]:.6f}\n")
        f.write(f"Best epoch: {history['best_epoch'][0]}\n")
        f.write(f"Best validation MSE: {history['best_val_mse'][0]:.10e}\n")
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