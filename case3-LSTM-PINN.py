

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import math
import random
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim



CONFIG = {

    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_root": "outputs_lstm_pinn_concentration_electric",


    "x_min": 0.0,
    "x_max": 1.0,
    "y_min": 0.0,
    "y_max": 1.0,


    "Re": 20.0,
    "Gamma_e": 1.2,
    "Gamma_T": 0.6,
    "Gamma_c": 0.4,
    "Pe_T": 15.0,
    "Omega_e": 0.25,
    "Pe_c": 18.0,
    "Xi": 0.30,


    "N_f": 3500,
    "N_b_each": 300,
    "N_d": 2800,


    "N_test_x": 201,
    "N_test_y": 201,
    "pred_chunk_size": 12000,


    "seq_len": 6,
    "lstm_input_dim": 1,
    "lstm_hidden_dim": 128,
    "lstm_num_layers": 2,
    "bidirectional": False,
    "dropout": 0.0,
    "fc_hidden_dim": 128,
    "num_fc_layers": 3,
    "output_dim": 5,
    "activation": "tanh",


    "epochs_adam": 5000,
    "lr": 1e-3,
    "weight_decay": 1e-8,
    "print_every": 100,
    "use_lbfgs": False,
    "lbfgs_max_iter": 180,


    "lr_patience": 200,
    "lr_factor": 0.7,
    "min_lr": 1e-6,


    "pde_points_per_epoch": 512,
    "val_pde_points": 256,
    "pde_batch_size": 32,
    "data_batch_size": 512,
    "bc_batch_size": 512,
    "empty_cache_every": 20,


    "lambda_pde": 1.0,
    "lambda_bc": 10.0,
    "lambda_data": 5.0,
}

PI = math.pi



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def prepare_dirs(cfg):
    root = cfg["save_root"]
    dirs = {
        "root": root,
        "models": os.path.join(root, "models"),
        "figures": os.path.join(root, "figures"),
        "txt": os.path.join(root, "txt"),
        "logs": os.path.join(root, "logs"),
        "metrics": os.path.join(root, "metrics"),
    }
    for p in dirs.values():
        ensure_dir(p)
    return dirs



def gradients(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False,
    )[0]


def second_gradients(outputs, inputs):
    first = gradients(outputs, inputs)
    second = gradients(first, inputs)
    return second


def batch_index_iterator(n: int, batch_size: int, shuffle: bool = False, seed: int = 42):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield idx[start:end]



def psi_exact(x, y):
    return (
        0.12
        * torch.exp(0.16 * x - 0.10 * y)
        * x * (1.0 - x) * y * (1.0 - y)
        * (
            torch.sin(PI * x + 0.23 * PI * y)
            + 0.18 * torch.cos(2.0 * PI * x * y + 0.11)
        )
    )


def T_exact(x, y):
    return (
        1.0
        + 0.18 * torch.exp(-0.12 * x + 0.08 * y)
        * (
            torch.cos(1.15 * PI * x * y + 0.20)
            + 0.35 * torch.sin(2.0 * PI * x + 0.31 * PI * y)
        )
        + 0.06 * x
    )


def p_exact(x, y):
    return (
        0.07 * torch.exp(0.07 * x + 0.05 * y)
        * (
            torch.cos(PI * x - 0.21 * PI * y)
            + 0.24 * torch.sin(2.0 * PI * x * y + 0.14)
        )
        + 0.02 * (x - y)
    )


def c_exact(x, y):
    return (
        0.95
        + 0.16 * torch.exp(0.09 * x - 0.13 * y)
        * (
            torch.sin(PI * x * y + 0.17)
            + 0.28 * torch.cos(1.70 * PI * y + 0.22 * PI * x)
        )
        + 0.04 * y
    )


def phi_e_exact(x, y):
    return (
        torch.exp(-0.08 * x)
        * (
            0.9 * torch.cos(PI * x + 0.18 * PI * y)
            + 0.12 * torch.sin(2.0 * PI * x * y)
        )
    )


def electric_field(x, y):
    phi_e = phi_e_exact(x, y)
    E_x = -gradients(phi_e, x)
    E_y = -gradients(phi_e, y)
    E2 = E_x ** 2 + E_y ** 2
    return phi_e, E_x, E_y, E2


def exact_solution(x, y):
    if (not x.requires_grad) or (not y.requires_grad):
        raise ValueError(
            "exact_solution() requires x and y with requires_grad=True "
            "because u,v and electric field use autograd."
        )

    psi = psi_exact(x, y)
    u = gradients(psi, y)
    v = -gradients(psi, x)

    T = T_exact(x, y)
    p = p_exact(x, y)
    c = c_exact(x, y)

    phi_e, E_x, E_y, E2 = electric_field(x, y)

    return {
        "psi": psi,
        "u": u,
        "v": v,
        "T": T,
        "p": p,
        "c": c,
        "phi_e": phi_e,
        "E_x": E_x,
        "E_y": E_y,
        "E2": E2,
    }


def exact_solution_tensor(x, y):
    sol = exact_solution(x, y)
    return torch.cat([sol["u"], sol["v"], sol["T"], sol["p"], sol["c"]], dim=1)


def exact_solution_tensor_detached(x, y):
    x_req = x.detach().clone().requires_grad_(True)
    y_req = y.detach().clone().requires_grad_(True)
    out = exact_solution_tensor(x_req, y_req)
    return out.detach()



def source_terms(x, y, cfg):
    Re = cfg["Re"]
    Gamma_e = cfg["Gamma_e"]
    Gamma_T = cfg["Gamma_T"]
    Gamma_c = cfg["Gamma_c"]
    Pe_T = cfg["Pe_T"]
    Omega_e = cfg["Omega_e"]
    Pe_c = cfg["Pe_c"]
    Xi = cfg["Xi"]

    sol = exact_solution(x, y)
    u = sol["u"]
    v = sol["v"]
    T = sol["T"]
    p = sol["p"]
    c = sol["c"]
    E_x = sol["E_x"]
    E_y = sol["E_y"]
    E2 = sol["E2"]

    u_x = gradients(u, x)
    u_y = gradients(u, y)
    v_x = gradients(v, x)
    v_y = gradients(v, y)

    T_x = gradients(T, x)
    T_y = gradients(T, y)
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    c_x = gradients(c, x)
    c_y = gradients(c, y)

    u_xx = second_gradients(u, x)
    u_yy = second_gradients(u, y)
    v_xx = second_gradients(v, x)
    v_yy = second_gradients(v, y)

    T_xx = second_gradients(T, x)
    T_yy = second_gradients(T, y)
    c_xx = second_gradients(c, x)
    c_yy = second_gradients(c, y)

    div_cE = gradients(c * E_x, x) + gradients(c * E_y, y)

    f_u = u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy) - Gamma_e * c * E_x
    f_v = u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy) - Gamma_e * c * E_y - Gamma_T * T - Gamma_c * c
    s_T = u * T_x + v * T_y - (1.0 / Pe_T) * (T_xx + T_yy) - Omega_e * c * E2
    s_c = u * c_x + v * c_y - (1.0 / Pe_c) * (c_xx + c_yy) + Xi * div_cE

    return {
        "f_u": f_u,
        "f_v": f_v,
        "s_T": s_T,
        "s_c": s_c,
    }


def source_terms_detached(x, y, cfg):
    x_req = x.detach().clone().requires_grad_(True)
    y_req = y.detach().clone().requires_grad_(True)
    src = source_terms(x_req, y_req, cfg)
    return {k: v.detach() for k, v in src.items()}



def sample_interior_points(n, cfg, device, requires_grad=False):
    x = np.random.rand(n, 1) * (cfg["x_max"] - cfg["x_min"]) + cfg["x_min"]
    y = np.random.rand(n, 1) * (cfg["y_max"] - cfg["y_min"]) + cfg["y_min"]

    x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=requires_grad)
    y = torch.tensor(y, dtype=torch.float32, device=device, requires_grad=requires_grad)
    return x, y


def sample_boundary_points(n_each, cfg, device):
    x_min, x_max = cfg["x_min"], cfg["x_max"]
    y_min, y_max = cfg["y_min"], cfg["y_max"]

    xb = np.random.rand(n_each, 1) * (x_max - x_min) + x_min
    yb = np.full((n_each, 1), y_min)

    xt = np.random.rand(n_each, 1) * (x_max - x_min) + x_min
    yt = np.full((n_each, 1), y_max)

    xl = np.full((n_each, 1), x_min)
    yl = np.random.rand(n_each, 1) * (y_max - y_min) + y_min

    xr = np.full((n_each, 1), x_max)
    yr = np.random.rand(n_each, 1) * (y_max - y_min) + y_min

    x = np.vstack([xb, xt, xl, xr])
    y = np.vstack([yb, yt, yl, yr])

    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    return x, y


def sample_supervised_data(n, cfg, device):
    x, y = sample_interior_points(n, cfg, device, requires_grad=False)
    target = exact_solution_tensor_detached(x, y)
    return x, y, target


def split_train_val(x, y, target, train_ratio=0.7):
    n = x.shape[0]
    idx = torch.randperm(n, device=x.device)
    n_train = int(train_ratio * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    return {
        "x_train": x[train_idx],
        "y_train": y[train_idx],
        "target_train": target[train_idx],
        "x_val": x[val_idx],
        "y_val": y[val_idx],
        "target_val": target[val_idx],
    }


def generate_test_grid(cfg, device):
    x = np.linspace(cfg["x_min"], cfg["x_max"], cfg["N_test_x"])
    y = np.linspace(cfg["y_min"], cfg["y_max"], cfg["N_test_y"])
    X, Y = np.meshgrid(x, y)

    XY = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    x_t = torch.tensor(XY[:, 0:1], dtype=torch.float32, device=device)
    y_t = torch.tensor(XY[:, 1:2], dtype=torch.float32, device=device)
    return X, Y, x_t, y_t



def get_activation(name):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class LSTMPINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.x_min = cfg["x_min"]
        self.x_max = cfg["x_max"]
        self.y_min = cfg["y_min"]
        self.y_max = cfg["y_max"]

        self.seq_len = cfg["seq_len"]
        self.lstm_input_dim = cfg["lstm_input_dim"]
        self.lstm_hidden_dim = cfg["lstm_hidden_dim"]
        self.lstm_num_layers = cfg["lstm_num_layers"]
        self.bidirectional = cfg["bidirectional"]
        self.dropout = cfg["dropout"]

        self.fc_hidden_dim = cfg["fc_hidden_dim"]
        self.num_fc_layers = cfg["num_fc_layers"]
        self.output_dim = cfg["output_dim"]
        self.activation_name = cfg["activation"]

        self.num_directions = 2 if self.bidirectional else 1
        lstm_dropout = self.dropout if self.lstm_num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=self.bidirectional
        )

        act = get_activation(self.activation_name)
        decoder_in_dim = self.lstm_hidden_dim * self.num_directions * 2

        decoder_layers = []
        in_dim = decoder_in_dim
        for _ in range(self.num_fc_layers):
            decoder_layers.append(nn.Linear(in_dim, self.fc_hidden_dim))
            decoder_layers.append(act)
            in_dim = self.fc_hidden_dim
        decoder_layers.append(nn.Linear(in_dim, self.output_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def normalize_xy(self, x, y):
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        return x_norm, y_norm

    def build_coordinate_sequence(self, x, y):
        x_n, y_n = self.normalize_xy(x, y)

        s1 = x_n
        s2 = y_n
        s3 = x_n + y_n
        s4 = x_n - y_n
        s5 = x_n * y_n
        s6 = x_n ** 2 + y_n ** 2

        seq = torch.stack([s1, s2, s3, s4, s5, s6], dim=1)
        return seq

    def forward(self, x, y):
        seq = self.build_coordinate_sequence(x, y)

        with torch.backends.cudnn.flags(enabled=False):
            lstm_out, _ = self.lstm(seq)

        last_feat = lstm_out[:, -1, :]
        mean_feat = lstm_out.mean(dim=1)
        features = torch.cat([last_feat, mean_feat], dim=1)

        out = self.decoder(features)
        return out



def split_fields(pred):
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    T = pred[:, 2:3]
    p = pred[:, 3:4]
    c = pred[:, 4:5]
    return u, v, T, p, c


def pde_residuals_from_tensors(model, x, y, f_u, f_v, s_T, s_c, cfg):
    Re = cfg["Re"]
    Gamma_e = cfg["Gamma_e"]
    Gamma_T = cfg["Gamma_T"]
    Gamma_c = cfg["Gamma_c"]
    Pe_T = cfg["Pe_T"]
    Omega_e = cfg["Omega_e"]
    Pe_c = cfg["Pe_c"]
    Xi = cfg["Xi"]

    pred = model(x, y)
    u, v, T, p, c = split_fields(pred)

    u_x = gradients(u, x)
    u_y = gradients(u, y)
    v_x = gradients(v, x)
    v_y = gradients(v, y)

    T_x = gradients(T, x)
    T_y = gradients(T, y)
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    c_x = gradients(c, x)
    c_y = gradients(c, y)

    u_xx = second_gradients(u, x)
    u_yy = second_gradients(u, y)
    v_xx = second_gradients(v, x)
    v_yy = second_gradients(v, y)

    T_xx = second_gradients(T, x)
    T_yy = second_gradients(T, y)
    c_xx = second_gradients(c, x)
    c_yy = second_gradients(c, y)

    _, E_x, E_y, E2 = electric_field(x, y)

    div_cE = gradients(c * E_x, x) + gradients(c * E_y, y)

    r_cont = u_x + v_y

    r_u = (
        u * u_x + v * u_y
        + p_x
        - (1.0 / Re) * (u_xx + u_yy)
        - Gamma_e * c * E_x
        - f_u
    )

    r_v = (
        u * v_x + v * v_y
        + p_y
        - (1.0 / Re) * (v_xx + v_yy)
        - Gamma_e * c * E_y
        - Gamma_T * T
        - Gamma_c * c
        - f_v
    )

    r_T = (
        u * T_x + v * T_y
        - (1.0 / Pe_T) * (T_xx + T_yy)
        - Omega_e * c * E2
        - s_T
    )

    r_c = (
        u * c_x + v * c_y
        - (1.0 / Pe_c) * (c_xx + c_yy)
        + Xi * div_cE
        - s_c
    )

    return r_cont, r_u, r_v, r_T, r_c



mse_loss = nn.MSELoss()


def compute_pde_loss_batched(model, batch, cfg, epoch_seed, n_use):
    x_all = batch["x_f"]
    y_all = batch["y_f"]
    f_u_all = batch["f_u"]
    f_v_all = batch["f_v"]
    s_T_all = batch["s_T"]
    s_c_all = batch["s_c"]

    n_all = x_all.shape[0]
    n_use = min(n_use, n_all)

    rng = np.random.default_rng(epoch_seed)
    chosen = rng.choice(n_all, size=n_use, replace=False)

    total_cont = 0.0
    total_u = 0.0
    total_v = 0.0
    total_T = 0.0
    total_c = 0.0
    nb = 0

    for idx in batch_index_iterator(n_use, cfg["pde_batch_size"], shuffle=False):
        chosen_idx = chosen[idx]

        x = x_all[chosen_idx].detach().clone().requires_grad_(True)
        y = y_all[chosen_idx].detach().clone().requires_grad_(True)
        f_u = f_u_all[chosen_idx]
        f_v = f_v_all[chosen_idx]
        s_T = s_T_all[chosen_idx]
        s_c = s_c_all[chosen_idx]

        r_cont, r_u, r_v, r_T, r_c = pde_residuals_from_tensors(model, x, y, f_u, f_v, s_T, s_c, cfg)

        loss_cont = torch.mean(r_cont ** 2)
        loss_u = torch.mean(r_u ** 2)
        loss_v = torch.mean(r_v ** 2)
        loss_T = torch.mean(r_T ** 2)
        loss_c = torch.mean(r_c ** 2)

        total_cont = total_cont + loss_cont
        total_u = total_u + loss_u
        total_v = total_v + loss_v
        total_T = total_T + loss_T
        total_c = total_c + loss_c
        nb += 1

        del x, y, f_u, f_v, s_T, s_c, r_cont, r_u, r_v, r_T, r_c, loss_cont, loss_u, loss_v, loss_T, loss_c

    loss_cont = total_cont / max(nb, 1)
    loss_u = total_u / max(nb, 1)
    loss_v = total_v / max(nb, 1)
    loss_T = total_T / max(nb, 1)
    loss_c = total_c / max(nb, 1)
    loss_pde = loss_cont + loss_u + loss_v + loss_T + loss_c

    detail = {
        "loss_cont": loss_cont,
        "loss_u": loss_u,
        "loss_v": loss_v,
        "loss_T": loss_T,
        "loss_c": loss_c,
    }
    return loss_pde, detail


def compute_bc_loss_batched(model, batch, cfg, use_val=False):
    x_key = "x_b"
    y_key = "y_b"
    t_key = "target_b"
    if use_val:
        x_key = "x_d_val"
        y_key = "y_d_val"
        t_key = "target_d_val"

    xy_n = batch["x_b"].shape[0]
    total = 0.0
    nb = 0
    for idx in batch_index_iterator(xy_n, cfg["bc_batch_size"], shuffle=False):
        pred_bc = model(batch["x_b"][idx], batch["y_b"][idx])
        target_bc = batch["target_b"][idx]
        total = total + mse_loss(pred_bc, target_bc)
        nb += 1
    return total / max(nb, 1)


def compute_data_loss_batched(model, batch, cfg, use_val=False, epoch_seed=42):
    x_key = "x_d_train"
    y_key = "y_d_train"
    t_key = "target_d_train"
    if use_val:
        x_key = "x_d_val"
        y_key = "y_d_val"
        t_key = "target_d_val"

    n = batch[x_key].shape[0]
    total = 0.0
    nb = 0
    for idx in batch_index_iterator(n, cfg["data_batch_size"], shuffle=not use_val, seed=epoch_seed):
        pred = model(batch[x_key][idx], batch[y_key][idx])
        total = total + mse_loss(pred, batch[t_key][idx])
        nb += 1
    return total / max(nb, 1)


def compute_supervised_mse(model, x, y, target, batch_size):
    model.eval()
    total = 0.0
    nb = 0
    with torch.no_grad():
        for idx in batch_index_iterator(x.shape[0], batch_size, shuffle=False):
            pred = model(x[idx], y[idx])
            total += torch.mean((pred - target[idx]) ** 2).item()
            nb += 1
    model.train()
    return total / max(nb, 1)


def compute_validation_details(model, batch, cfg):
    model.eval()
    val_pde, val_pde_detail = compute_pde_loss_batched(
        model, batch, cfg, epoch_seed=cfg["seed"] + 999999, n_use=cfg["val_pde_points"]
    )
    val_bc = compute_bc_loss_batched(model, batch, cfg)
    val_data = compute_data_loss_batched(model, batch, cfg, use_val=True, epoch_seed=cfg["seed"] + 777777)
    val_total = cfg["lambda_pde"] * val_pde + cfg["lambda_bc"] * val_bc + cfg["lambda_data"] * val_data
    model.train()
    return val_total, val_pde, val_bc, val_data, val_pde_detail


def compute_total_loss(model, batch, cfg, epoch_seed):
    loss_pde, pde_detail = compute_pde_loss_batched(
        model, batch, cfg, epoch_seed=epoch_seed, n_use=cfg["pde_points_per_epoch"]
    )
    loss_bc = compute_bc_loss_batched(model, batch, cfg)
    loss_data = compute_data_loss_batched(model, batch, cfg, use_val=False, epoch_seed=epoch_seed)

    total_loss = (
        cfg["lambda_pde"] * loss_pde
        + cfg["lambda_bc"] * loss_bc
        + cfg["lambda_data"] * loss_data
    )
    return total_loss, loss_pde, loss_bc, loss_data, pde_detail



def compute_metrics_one(pred, true, eps=1e-12):
    diff = pred - true
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    l2_rel = np.linalg.norm(diff.reshape(-1), 2) / (np.linalg.norm(true.reshape(-1), 2) + eps)
    return mse, rmse, mae, l2_rel


def compute_all_metrics(fields_pred, fields_true):
    rows = []
    for name in ["u", "v", "T", "p", "c"]:
        mse, rmse, mae, l2_rel = compute_metrics_one(fields_pred[name], fields_true[name])
        rows.append({
            "variable": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "L2_relative": l2_rel,
        })

    df = pd.DataFrame(rows)
    avg_row = {
        "variable": "average",
        "MSE": df["MSE"].mean(),
        "RMSE": df["RMSE"].mean(),
        "MAE": df["MAE"].mean(),
        "L2_relative": df["L2_relative"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    return df



def save_txt_field(X, Y, Z, path):
    data = np.column_stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])
    np.savetxt(path, data, fmt="%.10e", header="x y value", comments="")


def save_curve_txt(df, cols, path):
    df[cols].to_csv(path, index=False, sep="\t")



def plot_field(X, Y, Z, title, save_path, cmap="rainbow", levels=100):
    plt.figure(figsize=(7, 6))
    cf = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
    plt.colorbar(cf)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_combined_fields(X, Y, fields, names, title_prefix, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, name in enumerate(names):
        cf = axes[i].contourf(X, Y, fields[name], levels=100, cmap="rainbow")
        axes[i].set_title(f"{title_prefix}_{name}")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        fig.colorbar(cf, ax=axes[i])

    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_curves(log_df, figure_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(log_df["epoch"], log_df["total_loss"], label="total_loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "loss_total.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(log_df["epoch"], log_df["total_loss"], label="train_total_loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="val_loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "loss_train_val.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(log_df["epoch"], log_df["pde_loss"], label="pde_loss")
    plt.plot(log_df["epoch"], log_df["bc_loss"], label="bc_loss")
    plt.plot(log_df["epoch"], log_df["data_loss"], label="data_loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "loss_components.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(log_df["epoch"], log_df["loss_cont"], label="continuity")
    plt.plot(log_df["epoch"], log_df["loss_u"], label="momentum_u")
    plt.plot(log_df["epoch"], log_df["loss_v"], label="momentum_v")
    plt.plot(log_df["epoch"], log_df["loss_T"], label="energy")
    plt.plot(log_df["epoch"], log_df["loss_c"], label="concentration")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("PDE Sub Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "loss_pde_subcomponents.png"), dpi=300)
    plt.close()



def build_training_batch(cfg, device):

    x_f, y_f = sample_interior_points(cfg["N_f"], cfg, device, requires_grad=False)
    src = source_terms_detached(x_f, y_f, cfg)

    x_b, y_b = sample_boundary_points(cfg["N_b_each"], cfg, device)
    target_b = exact_solution_tensor_detached(x_b, y_b)

    x_d, y_d, target_d = sample_supervised_data(cfg["N_d"], cfg, device)
    split = split_train_val(x_d, y_d, target_d, train_ratio=0.7)

    batch = {
        "x_f": x_f.detach(),
        "y_f": y_f.detach(),
        "f_u": src["f_u"].detach(),
        "f_v": src["f_v"].detach(),
        "s_T": src["s_T"].detach(),
        "s_c": src["s_c"].detach(),

        "x_b": x_b.detach(),
        "y_b": y_b.detach(),
        "target_b": target_b.detach(),

        "x_d_train": split["x_train"].detach(),
        "y_d_train": split["y_train"].detach(),
        "target_d_train": split["target_train"].detach(),

        "x_d_val": split["x_val"].detach(),
        "y_d_val": split["y_val"].detach(),
        "target_d_val": split["target_val"].detach(),
    }
    return batch


def train(model, cfg, dirs):
    device = cfg["device"]

    t_data0 = time.time()
    batch = build_training_batch(cfg, device)
    data_generation_time = time.time() - t_data0

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["lr_factor"],
        patience=cfg["lr_patience"],
        min_lr=cfg["min_lr"]
    )

    best_model_path = os.path.join(dirs["models"], "best_model.pth")
    last_model_path = os.path.join(dirs["models"], "last_model.pth")

    log_rows = []
    best_val = float("inf")

    print(f"Using device: {device}")
    print("Training with Adam starts...")

    train_start_time = time.time()

    for epoch in range(1, cfg["epochs_adam"] + 1):
        model.train()
        epoch_start_time = time.time()

        optimizer.zero_grad(set_to_none=True)
        total_loss, loss_pde, loss_bc, loss_data, pde_detail = compute_total_loss(
            model, batch, cfg, epoch_seed=cfg["seed"] + epoch
        )
        total_loss.backward()
        optimizer.step()

        train_mse = compute_supervised_mse(
            model, batch["x_d_train"], batch["y_d_train"], batch["target_d_train"], cfg["data_batch_size"]
        )
        val_mse = compute_supervised_mse(
            model, batch["x_d_val"], batch["y_d_val"], batch["target_d_val"], cfg["data_batch_size"]
        )

        val_total, val_pde, val_bc, val_data, val_pde_detail = compute_validation_details(model, batch, cfg)
        scheduler.step(float(val_total.detach().item()))

        epoch_time = time.time() - epoch_start_time
        elapsed_train_time = time.time() - train_start_time

        row = {
            "epoch": epoch,
            "stage": "Adam",
            "total_loss": float(total_loss.detach().item()),
            "pde_loss": float(loss_pde.detach().item()),
            "bc_loss": float(loss_bc.detach().item()),
            "data_loss": float(loss_data.detach().item()),
            "val_loss": float(val_total.detach().item()),

            "loss_cont": float(pde_detail["loss_cont"].detach().item()),
            "loss_u": float(pde_detail["loss_u"].detach().item()),
            "loss_v": float(pde_detail["loss_v"].detach().item()),
            "loss_T": float(pde_detail["loss_T"].detach().item()),
            "loss_c": float(pde_detail["loss_c"].detach().item()),

            "val_pde_loss": float(val_pde.detach().item()),
            "val_bc_loss": float(val_bc.detach().item()),
            "val_data_loss": float(val_data.detach().item()),

            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
            "cumulative_time": elapsed_train_time,
        }
        log_rows.append(row)

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(model.state_dict(), best_model_path)

        if epoch == 1 or epoch % cfg["print_every"] == 0 or epoch == cfg["epochs_adam"]:
            print(
                f"Epoch [{epoch:5d}/{cfg['epochs_adam']}] | "
                f"Total={total_loss.detach().item():.6e} | "
                f"PDE={loss_pde.detach().item():.6e} | "
                f"Data={loss_data.detach().item():.6e} | "
                f"BC={loss_bc.detach().item():.6e} | "
                f"Mass={pde_detail['loss_cont'].detach().item():.6e} | "
                f"Ux={pde_detail['loss_u'].detach().item():.6e} | "
                f"Vy={pde_detail['loss_v'].detach().item():.6e} | "
                f"T={pde_detail['loss_T'].detach().item():.6e} | "
                f"C={pde_detail['loss_c'].detach().item():.6e} | "
                f"ValTotal={val_total.detach().item():.6e} | "
                f"ValPDE={val_pde.detach().item():.6e} | "
                f"ValData={val_data.detach().item():.6e} | "
                f"ValBC={val_bc.detach().item():.6e} | "
                f"TrainMSE={train_mse:.6e} | "
                f"ValMSE={val_mse:.6e} | "
                f"LR={optimizer.param_groups[0]['lr']:.3e} | "
                f"EpochTime={epoch_time:.3f}s | "
                f"TrainTime={elapsed_train_time:.3f}s"
            )

        if device == "cuda" and (epoch % cfg["empty_cache_every"] == 0):
            torch.cuda.empty_cache()

    if cfg["use_lbfgs"]:
        print("Training with L-BFGS starts...")

        optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=cfg["lbfgs_max_iter"],
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        start_epoch_idx = len(log_rows)

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss_lbfgs, _, _, _, _ = compute_total_loss(model, batch, cfg, epoch_seed=0)
            total_loss_lbfgs.backward()
            return total_loss_lbfgs

        lbfgs_start_time = time.time()
        optimizer_lbfgs.step(closure)
        lbfgs_time = time.time() - lbfgs_start_time
        elapsed_train_time = time.time() - train_start_time

        total_loss, loss_pde, loss_bc, loss_data, pde_detail = compute_total_loss(model, batch, cfg, epoch_seed=0)
        val_total, val_pde, val_bc, val_data, val_pde_detail = compute_validation_details(model, batch, cfg)

        train_mse = compute_supervised_mse(
            model, batch["x_d_train"], batch["y_d_train"], batch["target_d_train"], cfg["data_batch_size"]
        )
        val_mse = compute_supervised_mse(
            model, batch["x_d_val"], batch["y_d_val"], batch["target_d_val"], cfg["data_batch_size"]
        )

        row = {
            "epoch": start_epoch_idx + 1,
            "stage": "LBFGS",
            "total_loss": float(total_loss.detach().item()),
            "pde_loss": float(loss_pde.detach().item()),
            "bc_loss": float(loss_bc.detach().item()),
            "data_loss": float(loss_data.detach().item()),
            "val_loss": float(val_total.detach().item()),

            "loss_cont": float(pde_detail["loss_cont"].detach().item()),
            "loss_u": float(pde_detail["loss_u"].detach().item()),
            "loss_v": float(pde_detail["loss_v"].detach().item()),
            "loss_T": float(pde_detail["loss_T"].detach().item()),
            "loss_c": float(pde_detail["loss_c"].detach().item()),

            "val_pde_loss": float(val_pde.detach().item()),
            "val_bc_loss": float(val_bc.detach().item()),
            "val_data_loss": float(val_data.detach().item()),

            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "lr": np.nan,
            "epoch_time": lbfgs_time,
            "cumulative_time": elapsed_train_time,
        }
        log_rows.append(row)

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch [{start_epoch_idx + 1:5d}/{start_epoch_idx + 1}] | "
            f"Total={total_loss.detach().item():.6e} | "
            f"PDE={loss_pde.detach().item():.6e} | "
            f"Data={loss_data.detach().item():.6e} | "
            f"BC={loss_bc.detach().item():.6e} | "
            f"Mass={pde_detail['loss_cont'].detach().item():.6e} | "
            f"Ux={pde_detail['loss_u'].detach().item():.6e} | "
            f"Vy={pde_detail['loss_v'].detach().item():.6e} | "
            f"T={pde_detail['loss_T'].detach().item():.6e} | "
            f"C={pde_detail['loss_c'].detach().item():.6e} | "
            f"ValTotal={val_total.detach().item():.6e} | "
            f"ValPDE={val_pde.detach().item():.6e} | "
            f"ValData={val_data.detach().item():.6e} | "
            f"ValBC={val_bc.detach().item():.6e} | "
            f"TrainMSE={train_mse:.6e} | "
            f"ValMSE={val_mse:.6e} | "
            f"LR={1.0:.3e} | "
            f"EpochTime={lbfgs_time:.3f}s | "
            f"TrainTime={elapsed_train_time:.3f}s"
        )

    torch.save(model.state_dict(), last_model_path)

    log_df = pd.DataFrame(log_rows)
    training_log_csv = os.path.join(dirs["logs"], "training_log.csv")
    training_log_txt = os.path.join(dirs["logs"], "training_log.txt")
    log_df.to_csv(training_log_csv, index=False)
    with open(training_log_txt, "w", encoding="utf-8") as f:
        f.write(log_df.to_string(index=False))

    save_curve_txt(log_df, ["epoch", "total_loss"], os.path.join(dirs["txt"], "loss_total.txt"))
    save_curve_txt(log_df, ["epoch", "total_loss", "val_loss"], os.path.join(dirs["txt"], "loss_train_val.txt"))
    save_curve_txt(
        log_df,
        ["epoch", "pde_loss", "bc_loss", "data_loss", "loss_cont", "loss_u", "loss_v", "loss_T", "loss_c"],
        os.path.join(dirs["txt"], "loss_components.txt")
    )

    total_training_time = time.time() - train_start_time
    timing_info = {
        "data_generation_time": data_generation_time,
        "training_time": total_training_time,
    }

    return log_df, timing_info, best_model_path, last_model_path



def batched_predict(model, x_t, y_t, chunk_size=10000):
    preds = []
    n = x_t.shape[0]
    model.eval()
    with torch.no_grad():
        for i in range(0, n, chunk_size):
            xs = x_t[i:i + chunk_size]
            ys = y_t[i:i + chunk_size]
            pred = model(xs, ys)
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def batched_exact(x_t, y_t, chunk_size=10000):
    outs = []
    n = x_t.shape[0]
    for i in range(0, n, chunk_size):
        xs = x_t[i:i + chunk_size]
        ys = y_t[i:i + chunk_size]
        exact = exact_solution_tensor_detached(xs, ys)
        outs.append(exact.cpu().numpy())
    return np.vstack(outs)


def predict_on_grid(model, cfg, device):
    X, Y, x_t, y_t = generate_test_grid(cfg, device)

    pred = batched_predict(model, x_t, y_t, chunk_size=cfg["pred_chunk_size"])
    exact = batched_exact(x_t, y_t, chunk_size=cfg["pred_chunk_size"])

    fields_pred = {
        "u": pred[:, 0].reshape(cfg["N_test_y"], cfg["N_test_x"]),
        "v": pred[:, 1].reshape(cfg["N_test_y"], cfg["N_test_x"]),
        "T": pred[:, 2].reshape(cfg["N_test_y"], cfg["N_test_x"]),
        "p": pred[:, 3].reshape(cfg["N_test_y"], cfg["N_test_x"]),
        "c": pred[:, 4].reshape(cfg["N_test_y"], cfg["N_test_x"]),
    }

    fields_exact = {
        "u": exact[:, 0].reshape(cfg["N_test_y"], cfg["N_test_x"]),
        "v": exact[:, 1].reshape(cfg["N_test_y"], cfg["N_test_x"]),
        "T": exact[:, 2].reshape(cfg["N_test_y"], cfg["N_test_x"]),
        "p": exact[:, 3].reshape(cfg["N_test_y"], cfg["N_test_x"]),
        "c": exact[:, 4].reshape(cfg["N_test_y"], cfg["N_test_x"]),
    }

    fields_error = {
        k: np.abs(fields_pred[k] - fields_exact[k]) for k in fields_pred.keys()
    }

    return X, Y, fields_pred, fields_exact, fields_error



def save_all_outputs(X, Y, fields_pred, fields_exact, fields_error, dirs):
    names = ["u", "v", "T", "p", "c"]

    for name in names:
        plot_field(X, Y, fields_exact[name], f"{name}_exact", os.path.join(dirs["figures"], f"{name}_exact.png"))
        save_txt_field(X, Y, fields_exact[name], os.path.join(dirs["txt"], f"{name}_exact.txt"))

        plot_field(X, Y, fields_pred[name], f"{name}_pred", os.path.join(dirs["figures"], f"{name}_pred.png"))
        save_txt_field(X, Y, fields_pred[name], os.path.join(dirs["txt"], f"{name}_pred.txt"))

        plot_field(X, Y, fields_error[name], f"{name}_abs_error", os.path.join(dirs["figures"], f"{name}_abs_error.png"))
        save_txt_field(X, Y, fields_error[name], os.path.join(dirs["txt"], f"{name}_abs_error.txt"))

    plot_combined_fields(X, Y, fields_exact, names, "exact", os.path.join(dirs["figures"], "combined_exact.png"))
    plot_combined_fields(X, Y, fields_pred, names, "pred", os.path.join(dirs["figures"], "combined_pred.png"))
    plot_combined_fields(X, Y, fields_error, names, "abs_error", os.path.join(dirs["figures"], "combined_abs_error.png"))


def save_metrics(metrics_df, dirs):
    metrics_csv = os.path.join(dirs["metrics"], "metrics_summary.csv")
    metrics_txt = os.path.join(dirs["metrics"], "metrics_summary.txt")

    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(metrics_df.to_string(index=False))


def save_timing(timing_dict, dirs):
    timing_df = pd.DataFrame([timing_dict])
    timing_csv = os.path.join(dirs["metrics"], "timing.csv")
    timing_txt = os.path.join(dirs["metrics"], "timing.txt")

    timing_df.to_csv(timing_csv, index=False)
    with open(timing_txt, "w", encoding="utf-8") as f:
        f.write(timing_df.to_string(index=False))



def main():
    try:
        total_start = time.time()

        set_seed(CONFIG["seed"])
        dirs = prepare_dirs(CONFIG)
        device = CONFIG["device"]

        print(f"Using device: {device}")

        model = LSTMPINN(CONFIG).to(device)

        log_df, timing_info, best_model_path, last_model_path = train(model, CONFIG, dirs)

        plot_loss_curves(log_df, dirs["figures"])

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        t_test0 = time.time()
        X, Y, fields_pred, fields_exact, fields_error = predict_on_grid(model, CONFIG, device)
        test_time = time.time() - t_test0

        metrics_df = compute_all_metrics(fields_pred, fields_exact)
        save_metrics(metrics_df, dirs)

        save_all_outputs(X, Y, fields_pred, fields_exact, fields_error, dirs)

        total_runtime = time.time() - total_start

        timing_info.update({
            "inference_test_time": test_time,
            "total_runtime": total_runtime,
            "avg_epoch_time": float(log_df["epoch_time"].mean()) if len(log_df) > 0 else np.nan,
        })
        save_timing(timing_info, dirs)

        print("\nTraining finished successfully.")
        print(f"Best model saved to: {best_model_path}")
        print(f"Last model saved to: {last_model_path}")
        print(f"Training log saved to: {os.path.join(dirs['logs'], 'training_log.csv')}")
        print(f"Metrics saved to: {os.path.join(dirs['metrics'], 'metrics_summary.csv')}")
        print(f"Timing saved to: {os.path.join(dirs['metrics'], 'timing.csv')}")
        print(f"Figures saved to: {dirs['figures']}")
        print(f"TXT fields saved to: {dirs['txt']}")

        print("\nMetrics Summary:")
        print(metrics_df.to_string(index=False))

        print("\nTiming Summary:")
        for k, v in timing_info.items():
            print(f"{k}: {v:.6f}")

    except Exception as e:
        print("程序运行发生异常：")
        print(str(e))
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
