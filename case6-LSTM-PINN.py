

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


    L: float = 1.0
    h1: float = 0.5
    h2: float = 1.0
    x_e: float = 0.35
    beta: float = 25.0


    U0: float = 1.0
    T_w: float = 0.0
    dT: float = 1.0
    c_w: float = 0.0
    dc: float = 1.0
    p_out: float = 0.0
    P0: float = 1.0
    E0: float = 1.0


    Re: float = 40.0
    Pr: float = 1.0
    Sc: float = 1.0
    Pi_e: float = 0.5
    Pi_T: float = 0.1
    Pi_J: float = 0.05
    Lambda: float = 0.1


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


    seq_len: int = 6
    lstm_input_size: int = 1
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    bidirectional: bool = False


    fc_hidden_dim: int = 128
    fc_num_res_blocks: int = 4
    activation: str = "tanh"
    dropout: float = 0.0


    epochs: int = 5000
    lr: float = 1e-3
    weight_decay: float = 1e-8
    print_every: int = 100
    grad_clip: float = 10.0


    scheduler_step_size: int = 1000
    scheduler_gamma: float = 0.7


    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    lambda_data: float = 10.0


    use_lbfgs: bool = False
    lbfgs_max_iter: int = 200
    lbfgs_lr: float = 1.0


    plot_nx: int = 360
    plot_ny: int = 280
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




def h_channel(x: torch.Tensor, cfg: Config) -> torch.Tensor:

    return cfg.h1 + 0.5 * (cfg.h2 - cfg.h1) * (1.0 + torch.tanh(cfg.beta * (x - cfg.x_e)))


def dhdx_channel(x: torch.Tensor, cfg: Config) -> torch.Tensor:

    z = cfg.beta * (x - cfg.x_e)
    sech2 = 1.0 / torch.cosh(z) ** 2
    return 0.5 * (cfg.h2 - cfg.h1) * cfg.beta * sech2


def xi_eta(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    h = h_channel(x, cfg)
    dh = dhdx_channel(x, cfg)
    xi = x / cfg.L
    eta = y / h
    return xi, eta, h, dh




def A_field(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    return (
        1.0
        + 0.18 * torch.sin(pi * xi + 0.20)
        + 0.11 * torch.cos(1.5 * pi * xi) * torch.sin(pi * eta + 0.30)
        + 0.06 * xi * eta
    )


def A_eta_field(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    return 0.11 * pi * torch.cos(1.5 * pi * xi) * torch.cos(pi * eta + 0.30) + 0.06 * xi


def A_xi_field(xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    return (
        0.18 * pi * torch.cos(pi * xi + 0.20)
        - 0.165 * pi * torch.sin(1.5 * pi * xi) * torch.sin(pi * eta + 0.30)
        + 0.06 * eta
    )


def u_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:

    xi, eta, _, _ = xi_eta(x, y, cfg)
    A = A_field(xi, eta)
    A_eta = A_eta_field(xi, eta)
    return cfg.U0 * (
        -4.0 * eta * (1.0 - eta ** 2) * A
        + (1.0 - eta ** 2) ** 2 * A_eta
    )


def v_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:

    xi, eta, h, dh = xi_eta(x, y, cfg)
    A = A_field(xi, eta)
    A_eta = A_eta_field(xi, eta)
    A_xi = A_xi_field(xi, eta)

    F = (1.0 - eta ** 2) ** 2 * A
    F_eta = -4.0 * eta * (1.0 - eta ** 2) * A + (1.0 - eta ** 2) ** 2 * A_eta
    F_xi = (1.0 - eta ** 2) ** 2 * A_xi

    return -cfg.U0 * ((h / cfg.L) * F_xi + dh * (F - eta * F_eta))


def T_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:
    pi = math.pi
    xi, eta, _, _ = xi_eta(x, y, cfg)
    inside = (
        1.0
        + 0.23 * torch.sin(pi * xi + 0.10) * torch.cos(0.8 * pi * eta)
        + 0.09 * xi * eta
        + 0.05 * torch.cos(2.0 * pi * xi + pi * eta)
    )
    return cfg.T_w + cfg.dT * (1.0 - eta ** 2) * inside


def c_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:
    pi = math.pi
    xi, eta, _, _ = xi_eta(x, y, cfg)
    inside = (
        1.0
        + 0.19 * torch.cos(1.2 * pi * xi - 0.15) * torch.sin(0.9 * pi * eta + 0.25)
        + 0.08 * xi ** 2 * eta
        + 0.04 * torch.sin(pi * xi * eta + 0.30)
    )
    return cfg.c_w + cfg.dc * (1.0 - eta ** 2) * inside


def p_exact_formula(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:
    pi = math.pi
    xi, eta, _, _ = xi_eta(x, y, cfg)
    inside = (
        (1.0 - xi)
        + 0.14 * torch.sin(1.4 * pi * xi + 0.20) * (1.0 - 0.30 * eta)
        + 0.07 * torch.cos(pi * xi * eta + 0.40)
        + 0.03 * xi * eta ** 2
    )
    return cfg.p_out + cfg.P0 * inside


def electric_field(x: torch.Tensor, y: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    pi = math.pi
    xi, eta, _, _ = xi_eta(x, y, cfg)
    Ex = cfg.E0 * (1.0 + 0.12 * torch.sin(pi * xi))
    Ey = 0.08 * cfg.E0 * torch.cos(pi * xi) * torch.sin(0.5 * pi * eta + 0.20)
    E2 = Ex ** 2 + Ey ** 2
    return Ex, Ey, E2


def exact_solution(xy: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:

    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    x = xy[:, 0:1]
    y = xy[:, 1:2]

    u = u_exact_formula(x, y, cfg)
    v = v_exact_formula(x, y, cfg)
    T = T_exact_formula(x, y, cfg)
    p = p_exact_formula(x, y, cfg)
    c = c_exact_formula(x, y, cfg)
    Ex, Ey, E2 = electric_field(x, y, cfg)

    return {
        "u": u,
        "v": v,
        "T": T,
        "p": p,
        "c": c,
        "Ex": Ex,
        "Ey": Ey,
        "E2": E2,
    }




def compute_source_terms(xy: torch.Tensor, cfg: Config) -> Dict[str, torch.Tensor]:

    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)

    sol = exact_solution(xy, cfg)
    u, v, T, p, c = sol["u"], sol["v"], sol["T"], sol["p"], sol["c"]
    Ex, Ey, E2 = sol["Ex"], sol["Ey"], sol["E2"]

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
    div_cE = (
        safe_grad(cEx, xy, create_graph=True, retain_graph=True)[:, 0:1]
        + safe_grad(cEy, xy, create_graph=True, retain_graph=True)[:, 1:2]
    )

    f_u = u * u_x + v * u_y + p_x - (1.0 / cfg.Re) * lap_u - cfg.Pi_e * c * Ex
    f_v = u * v_x + v * v_y + p_y - (1.0 / cfg.Re) * lap_v - cfg.Pi_e * c * Ey - cfg.Pi_T * T
    f_T = u * T_x + v * T_y - (1.0 / (cfg.Re * cfg.Pr)) * lap_T - cfg.Pi_J * c * E2
    f_c = u * c_x + v * c_y - (1.0 / (cfg.Re * cfg.Sc)) * lap_c + cfg.Lambda * div_cE

    continuity_exact = u_x + v_y

    return {
        "f_u": f_u,
        "f_v": f_v,
        "f_T": f_T,
        "f_c": f_c,
        "continuity_exact": continuity_exact,
    }




def sample_interior_points(n: int, cfg: Config, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, cfg.L, size=(n, 1)).astype(np.float32)
    h = cfg.h1 + 0.5 * (cfg.h2 - cfg.h1) * (1.0 + np.tanh(cfg.beta * (x - cfg.x_e)))
    y = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32) * h.astype(np.float32)
    return np.hstack([x, y]).astype(np.float32)


def sample_boundary_points(cfg: Config, seed: int) -> Dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)
    n = cfg.n_boundary_each

    y_in = rng.uniform(-cfg.h1, cfg.h1, size=(n, 1)).astype(np.float32)
    inlet = np.hstack([np.zeros((n, 1), dtype=np.float32), y_in]).astype(np.float32)

    y_out = rng.uniform(-cfg.h2, cfg.h2, size=(n, 1)).astype(np.float32)
    outlet = np.hstack([np.full((n, 1), cfg.L, dtype=np.float32), y_out]).astype(np.float32)

    x_up = rng.uniform(0.0, cfg.L, size=(n, 1)).astype(np.float32)
    h_up = cfg.h1 + 0.5 * (cfg.h2 - cfg.h1) * (1.0 + np.tanh(cfg.beta * (x_up - cfg.x_e)))
    upper = np.hstack([x_up, h_up.astype(np.float32)]).astype(np.float32)

    x_low = rng.uniform(0.0, cfg.L, size=(n, 1)).astype(np.float32)
    h_low = cfg.h1 + 0.5 * (cfg.h2 - cfg.h1) * (1.0 + np.tanh(cfg.beta * (x_low - cfg.x_e)))
    lower = np.hstack([x_low, (-h_low).astype(np.float32)]).astype(np.float32)

    return {
        "inlet": inlet,
        "outlet": outlet,
        "upper": upper,
        "lower": lower,
        "all": np.vstack([inlet, outlet, upper, lower]).astype(np.float32)
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

    fu_all, fv_all, fT_all, fc_all = [], [], [], []
    for start in range(0, xy_np.shape[0], chunk_size):
        end = min(start + chunk_size, xy_np.shape[0])
        xy_chunk = to_tensor(xy_np[start:end], device=device, requires_grad=True)
        src = compute_source_terms(xy_chunk, cfg)
        fu_all.append(src["f_u"].detach().cpu().numpy())
        fv_all.append(src["f_v"].detach().cpu().numpy())
        fT_all.append(src["f_T"].detach().cpu().numpy())
        fc_all.append(src["f_c"].detach().cpu().numpy())
        del xy_chunk, src
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
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


def generate_plot_grid(cfg: Config) -> Dict[str, np.ndarray]:

    x = np.linspace(0.0, cfg.L, cfg.plot_nx, dtype=np.float32)
    y = np.linspace(-cfg.h2, cfg.h2, cfg.plot_ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    h = cfg.h1 + 0.5 * (cfg.h2 - cfg.h1) * (1.0 + np.tanh(cfg.beta * (X - cfg.x_e)))
    mask = np.abs(Y) <= h
    xy_valid = np.stack([X[mask], Y[mask]], axis=1).astype(np.float32)

    return {
        "X": X,
        "Y": Y,
        "mask": mask,
        "xy_valid": xy_valid,
    }


def generate_dataset(cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:


    xy_data = sample_interior_points(cfg.n_data_samples, cfg, seed=cfg.seed + 10)
    uvTpc_data = exact_solution_on_points_numpy(xy_data, cfg, device, cfg.exact_chunk_size)
    xy_data_tr, uvTpc_data_tr, xy_data_va, uvTpc_data_va = split_train_val(
        xy_data, uvTpc_data, cfg.train_ratio, seed=cfg.seed + 11
    )

    xy_col = sample_interior_points(cfg.n_collocation, cfg, seed=cfg.seed + 20)
    src_np = source_terms_on_points_numpy(xy_col, cfg, device, cfg.source_chunk_size)
    source_all = np.hstack([src_np["f_u"], src_np["f_v"], src_np["f_T"], src_np["f_c"]]).astype(np.float32)
    xy_col_tr, src_tr, xy_col_va, src_va = split_train_val(
        xy_col, source_all, cfg.train_ratio, seed=cfg.seed + 21
    )

    boundary_dict = sample_boundary_points(cfg, seed=cfg.seed + 30)
    xy_bc = boundary_dict["all"]
    uvTpc_bc = exact_solution_on_points_numpy(xy_bc, cfg, device, cfg.exact_chunk_size)
    xy_bc_tr, uvTpc_bc_tr, xy_bc_va, uvTpc_bc_va = split_train_val(
        xy_bc, uvTpc_bc, cfg.train_ratio, seed=cfg.seed + 31
    )

    plot_grid = generate_plot_grid(cfg)
    xy_plot_valid = plot_grid["xy_valid"]
    uvTpc_plot_valid = exact_solution_on_points_numpy(xy_plot_valid, cfg, device, cfg.exact_chunk_size)

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

        "plot_X": plot_grid["X"],
        "plot_Y": plot_grid["Y"],
        "plot_mask": plot_grid["mask"],
        "xy_plot_valid": to_tensor(xy_plot_valid, device=device, requires_grad=False),
        "uvTpc_plot_valid": to_tensor(uvTpc_plot_valid, device=device, requires_grad=False),
    }




class ResidualFCBlock(nn.Module):

    def __init__(self, hidden_dim: int, activation: str = "tanh", dropout: float = 0.0):
        super().__init__()
        if activation.lower() == "tanh":
            act_cls = nn.Tanh
        elif activation.lower() == "relu":
            act_cls = nn.ReLU
        else:
            raise ValueError("Only tanh or relu is supported.")

        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_cls(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            act_cls(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LSTMPINN(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.num_directions = 2 if cfg.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=cfg.lstm_input_size,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional
        )

        fused_dim = cfg.lstm_hidden_size * self.num_directions + 2

        if cfg.activation.lower() == "tanh":
            act_cls = nn.Tanh
        elif cfg.activation.lower() == "relu":
            act_cls = nn.ReLU
        else:
            raise ValueError("Only tanh or relu is supported.")

        self.input_fusion = nn.Sequential(
            nn.Linear(fused_dim, cfg.fc_hidden_dim),
            act_cls()
        )

        self.res_blocks = nn.ModuleList([
            ResidualFCBlock(cfg.fc_hidden_dim, activation=cfg.activation, dropout=cfg.dropout)
            for _ in range(cfg.fc_num_res_blocks)
        ])

        self.output_layer = nn.Linear(cfg.fc_hidden_dim, cfg.output_dim)

        self.apply(xavier_init)
        self._init_lstm_parameters()

    def _init_lstm_parameters(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def build_sequence(self, xy: torch.Tensor) -> torch.Tensor:

        x = xy[:, 0:1]
        y = xy[:, 1:2]
        xi, eta, _, _ = xi_eta(x, y, self.cfg)

        ones = torch.ones_like(x)
        seq = torch.stack([x, y, xi, eta, x * y, ones], dim=1)  # (batch, 6, 1)
        return seq

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        seq = self.build_sequence(xy)

        # Disable CuDNN for double backward support
        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, _ = self.lstm(seq)

        last_feat = lstm_out[:, -1, :]
        fused = torch.cat([last_feat, xy], dim=1)

        z = self.input_fusion(fused)
        for block in self.res_blocks:
            z = block(z)

        out = self.output_layer(z)
        return out


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    return LSTMPINN(cfg).to(device)




def split_src(src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return src[:, 0:1], src[:, 1:2], src[:, 2:3], src[:, 3:4]


def pde_residuals(
    model: nn.Module,
    xy: torch.Tensor,
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

    x = xy[:, 0:1]
    y = xy[:, 1:2]
    Ex, Ey, E2 = electric_field(x, y, cfg)

    cEx = c * Ex
    cEy = c * Ey
    div_cE = (
        safe_grad(cEx, xy, create_graph=True, retain_graph=True)[:, 0:1]
        + safe_grad(cEy, xy, create_graph=True, retain_graph=True)[:, 1:2]
    )

    r_cont = u_x + v_y
    r_u = u * u_x + v * u_y + p_x - (1.0 / cfg.Re) * lap_u - cfg.Pi_e * c * Ex - f_u
    r_v = u * v_x + v * v_y + p_y - (1.0 / cfg.Re) * lap_v - cfg.Pi_e * c * Ey - cfg.Pi_T * T - f_v
    r_T = u * T_x + v * T_y - (1.0 / (cfg.Re * cfg.Pr)) * lap_T - cfg.Pi_J * c * E2 - f_T
    r_c = u * c_x + v * c_y - (1.0 / (cfg.Re * cfg.Sc)) * lap_c + cfg.Lambda * div_cE - f_c

    return {
        "continuity": r_cont,
        "momentum_u": r_u,
        "momentum_v": r_v,
        "temperature": r_T,
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

    total_cont = 0.0
    total_u = 0.0
    total_v = 0.0
    total_T = 0.0
    total_c = 0.0
    nb = 0

    for idx in batch_iterator(n_use, cfg.pde_batch_size, shuffle=False):
        xy_batch = xy_use[idx].clone().detach().requires_grad_(True)
        f_u, f_v, f_T, f_c = split_src(src_use[idx])

        res = pde_residuals(model, xy_batch, f_u, f_v, f_T, f_c, cfg)

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

        del xy_batch, f_u, f_v, f_T, f_c, res, loss_cont, loss_u, loss_v, loss_T, loss_c

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

    loss_total = cfg.lambda_pde * pde_losses["loss_pde"] + cfg.lambda_bc * loss_bc + cfg.lambda_data * loss_data

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
    total_val = cfg.lambda_pde * loss_pde_val + cfg.lambda_bc * loss_bc_val + cfg.lambda_data * loss_data_val

    return {
        "val_total_loss": total_val,
        "val_pde_loss": loss_pde_val,
        "val_bc_loss": loss_bc_val,
        "val_data_loss": loss_data_val,
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
    """Save x y value as three columns."""
    data = np.column_stack([x.reshape(-1), y.reshape(-1), value.reshape(-1)])
    np.savetxt(filepath, data, fmt="%.8f %.8f %.8f")




def plot_field_with_walls(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                          x_line: np.ndarray, y_up: np.ndarray, y_low: np.ndarray,
                          title: str, save_path: str, dpi: int = 300) -> None:
    plt.figure(figsize=(8, 5))
    contour = plt.contourf(X, Y, Z, levels=120, cmap="rainbow")
    plt.colorbar(contour)
    plt.plot(x_line, y_up, "k-", linewidth=1.2)
    plt.plot(x_line, y_low, "k-", linewidth=1.2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def build_plot_fields(valid_values: np.ndarray, X: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill full grid with NaN outside the physical domain."""
    Z = np.full_like(X, np.nan, dtype=np.float32)
    Z[mask] = valid_values.reshape(-1).astype(np.float32)
    return Z


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
    plt.plot(history["loss_bc"], label="train bc")
    plt.plot(history["loss_data"], label="train data")
    plt.plot(history["val_pde_loss"], label="val pde")
    plt.plot(history["val_bc_loss"], label="val bc")
    plt.plot(history["val_data_loss"], label="val data")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Components History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_components_history.png"), dpi=dpi)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(history["loss_cont"], label="continuity")
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


def compute_plot_predictions(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config):
    pred_valid = predict_in_batches(model, dataset["xy_plot_valid"], batch_size=cfg.predict_chunk_size)
    true_valid = dataset["uvTpc_plot_valid"].detach().cpu().numpy()
    return pred_valid, true_valid


def plot_all_results(model: nn.Module, dataset: Dict[str, torch.Tensor], cfg: Config, dirs: Dict[str, str]) -> None:

    pred_valid, true_valid = compute_plot_predictions(model, dataset, cfg)

    X = dataset["plot_X"]
    Y = dataset["plot_Y"]
    mask = dataset["plot_mask"]

    x_line = np.linspace(0.0, cfg.L, cfg.plot_nx, dtype=np.float32)
    h_line = cfg.h1 + 0.5 * (cfg.h2 - cfg.h1) * (1.0 + np.tanh(cfg.beta * (x_line - cfg.x_e)))
    y_up = h_line
    y_low = -h_line

    valid_xy = dataset["xy_plot_valid"].detach().cpu().numpy()
    xv = valid_xy[:, 0]
    yv = valid_xy[:, 1]

    names = ["u", "v", "T", "p", "c"]

    for i, name in enumerate(names):
        pred_field = build_plot_fields(pred_valid[:, i], X, Y, mask)
        true_field = build_plot_fields(true_valid[:, i], X, Y, mask)
        abs_err_field = build_plot_fields(np.abs(pred_valid[:, i] - true_valid[:, i]), X, Y, mask)

        plot_field_with_walls(
            X, Y, pred_field, x_line, y_up, y_low,
            f"{name} Prediction",
            os.path.join(dirs["figures"], f"{name}_pred.png"),
            dpi=cfg.dpi
        )
        plot_field_with_walls(
            X, Y, true_field, x_line, y_up, y_low,
            f"{name} True",
            os.path.join(dirs["figures"], f"{name}_true.png"),
            dpi=cfg.dpi
        )
        plot_field_with_walls(
            X, Y, abs_err_field, x_line, y_up, y_low,
            f"{name} Absolute Error",
            os.path.join(dirs["figures"], f"{name}_abs_error.png"),
            dpi=cfg.dpi
        )

        save_txt_xyz(os.path.join(dirs["txt"], f"{name}_pred.txt"), xv, yv, pred_valid[:, i])
        save_txt_xyz(os.path.join(dirs["txt"], f"{name}_true.txt"), xv, yv, true_valid[:, i])
        save_txt_xyz(os.path.join(dirs["txt"], f"{name}_abs_error.txt"), xv, yv, np.abs(pred_valid[:, i] - true_valid[:, i]))


    model.eval()
    xy_valid = dataset["xy_plot_valid"].clone().detach()
    xy_np = xy_valid.detach().cpu().numpy()
    src_np = source_terms_on_points_numpy(xy_np, cfg, xy_valid.device, cfg.source_chunk_size)
    src_tensor = np.hstack([src_np["f_u"], src_np["f_v"], src_np["f_T"], src_np["f_c"]]).astype(np.float32)
    src_tensor = to_tensor(src_tensor, xy_valid.device, requires_grad=False)

    r_cont_all, r_u_all, r_v_all, r_T_all, r_c_all = [], [], [], [], []
    n = xy_valid.shape[0]
    for idx in batch_iterator(n, cfg.pde_batch_size, shuffle=False):
        xy_batch = xy_valid[idx].clone().detach().requires_grad_(True)
        f_u, f_v, f_T, f_c = split_src(src_tensor[idx])
        res = pde_residuals(model, xy_batch, f_u, f_v, f_T, f_c, cfg)
        r_cont_all.append(res["continuity"].detach().cpu().numpy())
        r_u_all.append(res["momentum_u"].detach().cpu().numpy())
        r_v_all.append(res["momentum_v"].detach().cpu().numpy())
        r_T_all.append(res["temperature"].detach().cpu().numpy())
        r_c_all.append(res["concentration"].detach().cpu().numpy())
        del xy_batch, f_u, f_v, f_T, f_c, res

    residual_dict = {
        "r_continuity": np.vstack(r_cont_all).reshape(-1),
        "r_momentum_u": np.vstack(r_u_all).reshape(-1),
        "r_momentum_v": np.vstack(r_v_all).reshape(-1),
        "r_temperature": np.vstack(r_T_all).reshape(-1),
        "r_concentration": np.vstack(r_c_all).reshape(-1),
    }

    for rname, rval in residual_dict.items():
        rfield = build_plot_fields(rval, X, Y, mask)
        plot_field_with_walls(
            X, Y, rfield, x_line, y_up, y_low,
            rname,
            os.path.join(dirs["figures"], f"{rname}.png"),
            dpi=cfg.dpi
        )
        save_txt_xyz(os.path.join(dirs["txt"], f"{rname}.txt"), xv, yv, rval)




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
        "loss_bc": [],
        "loss_data": [],
        "loss_cont": [],
        "loss_u": [],
        "loss_v": [],
        "loss_T": [],
        "loss_c": [],
        "val_total_loss": [],
        "val_pde_loss": [],
        "val_bc_loss": [],
        "val_data_loss": [],
        "train_mse": [],
        "val_mse": [],
        "epoch_time": [],
        "train_time": [],
        "lr": [],
    }

    with open(os.path.join(dirs["logs"], "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger.log(f"Device: {device}")
    logger.log(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.log(f"Geometry: L={cfg.L}, h1={cfg.h1}, h2={cfg.h2}, xe={cfg.x_e}, beta={cfg.beta}")
    logger.log(f"Supervised points: {cfg.n_data_samples}")
    logger.log(f"Collocation points: {cfg.n_collocation}")
    logger.log(f"Boundary points total: {4 * cfg.n_boundary_each}")
    logger.log(f"Train ratio: {cfg.train_ratio}")
    logger.log(f"Epochs: {cfg.epochs}")
    logger.log(f"Learning rate: {cfg.lr}")
    logger.log("Model: LSTM-PINN")
    logger.log("-" * 120)

    train_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

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
        epoch_time = time.time() - epoch_start
        elapsed_train_time = time.time() - train_start

        history["loss_total"].append(losses["loss_total"].item())
        history["loss_pde"].append(losses["loss_pde"].item())
        history["loss_bc"].append(losses["loss_bc"].item())
        history["loss_data"].append(losses["loss_data"].item())
        history["loss_cont"].append(losses["loss_cont"].item())
        history["loss_u"].append(losses["loss_u"].item())
        history["loss_v"].append(losses["loss_v"].item())
        history["loss_T"].append(losses["loss_T"].item())
        history["loss_c"].append(losses["loss_c"].item())
        history["val_total_loss"].append(val_losses["val_total_loss"])
        history["val_pde_loss"].append(val_losses["val_pde_loss"])
        history["val_bc_loss"].append(val_losses["val_bc_loss"])
        history["val_data_loss"].append(val_losses["val_data_loss"])
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["epoch_time"].append(epoch_time)
        history["train_time"].append(elapsed_train_time)
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
                f"Mass={losses['loss_cont'].item():.6e} | "
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
                f"EpochTime={epoch_time:.3f}s | "
                f"TrainTime={elapsed_train_time:.3f}s"
            )

        if device.type == "cuda" and (epoch % cfg.empty_cache_every == 0):
            torch.cuda.empty_cache()

    if cfg.use_lbfgs:
        logger.log("Starting optional LBFGS refinement.")

        lbfgs = optim.LBFGS(
            model.parameters(),
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter
        )

        def closure():
            lbfgs.zero_grad()
            lbfgs_losses = compute_train_losses(model, dataset, cfg, epoch=0)
            lbfgs_losses["loss_total"].backward()
            return lbfgs_losses["loss_total"]

        lbfgs_start = time.time()
        lbfgs.step(closure)
        lbfgs_epoch_time = time.time() - lbfgs_start
        elapsed_train_time = time.time() - train_start

        lbfgs_losses = compute_train_losses(model, dataset, cfg, epoch=0)
        val_losses = compute_val_losses(model, dataset, cfg)

        train_mse = compute_supervised_mse_batched(
            model, dataset["xy_data_train"], dataset["uvTpc_data_train"], cfg.data_batch_size
        )
        val_mse = compute_supervised_mse_batched(
            model, dataset["xy_data_val"], dataset["uvTpc_data_val"], cfg.data_batch_size
        )

        history["loss_total"].append(lbfgs_losses["loss_total"].item())
        history["loss_pde"].append(lbfgs_losses["loss_pde"].item())
        history["loss_bc"].append(lbfgs_losses["loss_bc"].item())
        history["loss_data"].append(lbfgs_losses["loss_data"].item())
        history["loss_cont"].append(lbfgs_losses["loss_cont"].item())
        history["loss_u"].append(lbfgs_losses["loss_u"].item())
        history["loss_v"].append(lbfgs_losses["loss_v"].item())
        history["loss_T"].append(lbfgs_losses["loss_T"].item())
        history["loss_c"].append(lbfgs_losses["loss_c"].item())
        history["val_total_loss"].append(val_losses["val_total_loss"])
        history["val_pde_loss"].append(val_losses["val_pde_loss"])
        history["val_bc_loss"].append(val_losses["val_bc_loss"])
        history["val_data_loss"].append(val_losses["val_data_loss"])
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["epoch_time"].append(lbfgs_epoch_time)
        history["train_time"].append(elapsed_train_time)
        history["lr"].append(cfg.lbfgs_lr)

        if val_losses["val_total_loss"] < best_val_loss:
            best_val_loss = val_losses["val_total_loss"]
            best_epoch = cfg.epochs
            torch.save(model.state_dict(), best_model_path)

        logger.log(
            f"Epoch [{cfg.epochs + 1:5d}/{cfg.epochs + 1}] | "
            f"Total={lbfgs_losses['loss_total'].item():.6e} | "
            f"PDE={lbfgs_losses['loss_pde'].item():.6e} | "
            f"Data={lbfgs_losses['loss_data'].item():.6e} | "
            f"BC={lbfgs_losses['loss_bc'].item():.6e} | "
            f"Mass={lbfgs_losses['loss_cont'].item():.6e} | "
            f"Ux={lbfgs_losses['loss_u'].item():.6e} | "
            f"Vy={lbfgs_losses['loss_v'].item():.6e} | "
            f"T={lbfgs_losses['loss_T'].item():.6e} | "
            f"C={lbfgs_losses['loss_c'].item():.6e} | "
            f"ValTotal={val_losses['val_total_loss']:.6e} | "
            f"ValPDE={val_losses['val_pde_loss']:.6e} | "
            f"ValData={val_losses['val_data_loss']:.6e} | "
            f"ValBC={val_losses['val_bc_loss']:.6e} | "
            f"TrainMSE={train_mse:.6e} | "
            f"ValMSE={val_mse:.6e} | "
            f"LR={cfg.lbfgs_lr:.3e} | "
            f"EpochTime={lbfgs_epoch_time:.3f}s | "
            f"TrainTime={elapsed_train_time:.3f}s"
        )

    train_end = time.time()
    total_training_time = train_end - train_start

    torch.save(model.state_dict(), final_model_path)

    history_csv = os.path.join(dirs["logs"], "loss_history.csv")
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "loss_total", "loss_pde", "loss_bc", "loss_data",
            "loss_cont", "loss_u", "loss_v", "loss_T", "loss_c",
            "val_total_loss", "val_pde_loss", "val_bc_loss", "val_data_loss",
            "train_mse", "val_mse", "epoch_time", "train_time", "lr"
        ])
        for i in range(len(history["loss_total"])):
            writer.writerow([
                i + 1,
                history["loss_total"][i],
                history["loss_pde"][i],
                history["loss_bc"][i],
                history["loss_data"][i],
                history["loss_cont"][i],
                history["loss_u"][i],
                history["loss_v"][i],
                history["loss_T"][i],
                history["loss_c"][i],
                history["val_total_loss"][i],
                history["val_pde_loss"][i],
                history["val_bc_loss"][i],
                history["val_data_loss"][i],
                history["train_mse"][i],
                history["val_mse"][i],
                history["epoch_time"][i],
                history["train_time"][i],
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
    plot_metrics = evaluate_on_points(model, dataset["xy_plot_valid"], dataset["uvTpc_plot_valid"], CFG)

    print_metrics("Training-set Metrics", train_metrics)
    print_metrics("Validation-set Metrics", val_metrics)
    print_metrics("Plot-grid Metrics", plot_metrics)

    save_metrics(train_metrics, os.path.join(dirs["logs"], "metrics_train"))
    save_metrics(val_metrics, os.path.join(dirs["logs"], "metrics_val"))
    save_metrics(plot_metrics, os.path.join(dirs["logs"], "metrics_plot_grid"))

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