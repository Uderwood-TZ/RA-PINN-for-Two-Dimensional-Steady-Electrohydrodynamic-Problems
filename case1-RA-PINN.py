

import os
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
    "save_root": "results_electrothermal_residual_attention_pinn",


    "x_min": 0.0,
    "x_max": 1.0,
    "y_min": 0.0,
    "y_max": 1.0,


    "nu": 0.01,
    "alpha": 0.01,
    "D": 0.01,
    "lambda_1": 0.5,
    "lambda_2": 0.4,
    "beta_T": 0.2,
    "beta_c": 0.1,
    "gamma_1": 0.05,
    "gamma_2": 0.03,


    "N_f": 4000,
    "N_bc_each": 300,
    "N_data": 3000,


    "N_test_x": 201,
    "N_test_y": 201,


    "input_dim": 2,
    "output_dim": 5,
    "hidden_dim": 128,
    "num_blocks": 6,
    "activation": "tanh",
    "attention_reduction": 4,
    "attention_positions": [1, 3, 5],


    "epochs": 5000,
    "lr": 1e-3,
    "scheduler_step": 800,
    "scheduler_gamma": 0.7,
    "print_every": 100,
    "weight_decay": 1e-8,


    "lambda_pde": 1.0,
    "lambda_bc": 10.0,
    "lambda_data": 5.0,


    "attention_eta": 1.0,
    "attention_eps": 1e-12,


    "use_lbfgs": False,
    "lbfgs_max_iter": 200,
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
        "figures": os.path.join(root, "figures"),
        "txt": os.path.join(root, "txt"),
        "models": os.path.join(root, "models"),
        "logs": os.path.join(root, "logs"),
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
    )[0]


def second_gradients(outputs, inputs):

    first = gradients(outputs, inputs)
    second = gradients(first, inputs)
    return second



def psi_exact(x, y):

    return (
        torch.exp(0.11 * x - 0.07 * y)
        * x**2 * (1.0 - x)**2
        * y**2 * (1.0 - y)**2
        * (
            torch.sin(PI * (x + 0.18 * y**2 + 0.07 * x * y))
            + 0.31 * torch.cos(PI * (1.15 * y + 0.09 * x**2))
        )
    )


def T_exact(x, y):

    return (
        1.0
        + 0.28 * torch.exp(0.05 * x + 0.03 * y) * x * (1.0 - x) * y * (1.0 - y)
        * (
            0.73 * torch.cos(PI * (0.92 * x + 0.16 * y**2))
            + 0.27 * torch.sin(PI * (1.08 * y + 0.11 * x**2 + 0.09 * x * y))
        )
    )


def p_raw(x, y):

    return (
        torch.exp(0.04 * x - 0.05 * y)
        * (
            torch.cos(PI * (0.78 * x + 0.12 * y**2 + 0.06 * x * y))
            + 0.35 * torch.sin(PI * (1.18 * y + 0.07 * x**2))
        )
    )


def get_p_ref(device):

    x0 = torch.tensor([[0.5]], dtype=torch.float32, device=device)
    y0 = torch.tensor([[0.5]], dtype=torch.float32, device=device)
    with torch.no_grad():
        pref = p_raw(x0, y0)
    return pref


def p_exact(x, y, p_ref):

    return p_raw(x, y) - p_ref


def c_exact(x, y):

    return (
        1.0
        + 0.22 * torch.exp(0.06 * x + 0.04 * y) * x * (1.0 - x) * y * (1.0 - y)
        * (
            torch.sin(PI * (0.87 * x**2 + y + 0.08 * x * y))
            + 0.21 * torch.cos(PI * (2.0 * y - 0.30 * x))
        )
    )


def exact_solution(x, y, p_ref):

    if (not x.requires_grad) or (not y.requires_grad):
        raise ValueError(
            "exact_solution() requires x and y with requires_grad=True "
            "because u and v are obtained from psi by autograd."
        )

    psi = psi_exact(x, y)
    u = gradients(psi, y)
    v = -gradients(psi, x)

    T = T_exact(x, y)
    p = p_exact(x, y, p_ref)
    c = c_exact(x, y)

    return {
        "psi": psi,
        "u": u,
        "v": v,
        "T": T,
        "p": p,
        "c": c,
    }


def exact_solution_tensor(x, y, p_ref):

    sol = exact_solution(x, y, p_ref)
    return torch.cat([sol["u"], sol["v"], sol["T"], sol["p"], sol["c"]], dim=1)


def exact_solution_tensor_detached(x, y, p_ref):

    x_req = x.detach().clone().requires_grad_(True)
    y_req = y.detach().clone().requires_grad_(True)
    target = exact_solution_tensor(x_req, y_req, p_ref)
    return target.detach()



def source_terms(x, y, cfg, p_ref):


    nu = cfg["nu"]
    alpha = cfg["alpha"]
    D = cfg["D"]
    lambda_1 = cfg["lambda_1"]
    lambda_2 = cfg["lambda_2"]
    beta_T = cfg["beta_T"]
    beta_c = cfg["beta_c"]
    gamma_1 = cfg["gamma_1"]
    gamma_2 = cfg["gamma_2"]

    sol = exact_solution(x, y, p_ref)
    u = sol["u"]
    v = sol["v"]
    T = sol["T"]
    p = sol["p"]
    c = sol["c"]


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


    div_c_grad_T = gradients(c * T_x, x) + gradients(c * T_y, y)


    f_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy) - lambda_1 * c * T_x
    f_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy) - lambda_2 * c * T_y - beta_T * T - beta_c * c
    Q = u * T_x + v * T_y - alpha * (T_xx + T_yy) - gamma_1 * (T_x**2 + T_y**2)
    S = u * c_x + v * c_y - D * (c_xx + c_yy) - gamma_2 * div_c_grad_T

    return {
        "f_u": f_u,
        "f_v": f_v,
        "Q": Q,
        "S": S,
    }


def source_terms_detached(x, y, cfg, p_ref):

    x_req = x.detach().clone().requires_grad_(True)
    y_req = y.detach().clone().requires_grad_(True)
    src = source_terms(x_req, y_req, cfg, p_ref)
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


def sample_supervised_data(n, cfg, device, p_ref):

    x, y = sample_interior_points(n, cfg, device, requires_grad=False)
    target = exact_solution_tensor_detached(x, y, p_ref)
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
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class ResidualBlock(nn.Module):

    def __init__(self, dim, activation="tanh"):
        super().__init__()
        act = get_activation(activation)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = act

    def forward(self, x):
        identity = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out = self.act(identity + out)
        return out


class FeatureAttentionBlock(nn.Module):

    def __init__(self, dim, reduction=4, activation="tanh"):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = get_activation(activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.sigmoid(self.fc2(self.act(self.fc1(x))))
        return x * gate


class ResidualAttentionPINN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        input_dim = cfg["input_dim"]
        output_dim = cfg["output_dim"]
        hidden_dim = cfg["hidden_dim"]
        num_blocks = cfg["num_blocks"]
        activation = cfg["activation"]
        reduction = cfg["attention_reduction"]
        attention_positions = set(cfg["attention_positions"])

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(activation),
        )

        self.blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleDict()

        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(hidden_dim, activation=activation))
            if i in attention_positions:
                self.attn_blocks[str(i)] = FeatureAttentionBlock(
                    hidden_dim, reduction=reduction, activation=activation
                )

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        h = self.input_layer(inp)

        for i, block in enumerate(self.blocks):
            h = block(h)
            if str(i) in self.attn_blocks:
                h = self.attn_blocks[str(i)](h)

        out = self.output_layer(h)
        return out



def residual_attention_weights(r_cont, r_u, r_v, r_T, r_c, eta=1.0, eps=1e-12, detach=True):

    r_total = torch.sqrt(r_cont**2 + r_u**2 + r_v**2 + r_T**2 + r_c**2 + eps)

    if detach:
        r_used = r_total.detach()
    else:
        r_used = r_total

    w = 1.0 + eta * r_used / (torch.mean(r_used) + eps)
    return w, r_total



def split_fields(pred):

    u = pred[:, 0:1]
    v = pred[:, 1:2]
    T = pred[:, 2:3]
    p = pred[:, 3:4]
    c = pred[:, 4:5]
    return u, v, T, p, c


def pde_residuals(model, x, y, cfg, batch):

    nu = cfg["nu"]
    alpha = cfg["alpha"]
    D = cfg["D"]
    lambda_1 = cfg["lambda_1"]
    lambda_2 = cfg["lambda_2"]
    beta_T = cfg["beta_T"]
    beta_c = cfg["beta_c"]
    gamma_1 = cfg["gamma_1"]
    gamma_2 = cfg["gamma_2"]

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


    div_c_grad_T = gradients(c * T_x, x) + gradients(c * T_y, y)


    f_u = batch["f_u"]
    f_v = batch["f_v"]
    Q = batch["Q"]
    S = batch["S"]


    r_cont = u_x + v_y

    r_u = (
        u * u_x + v * u_y
        + p_x
        - nu * (u_xx + u_yy)
        - lambda_1 * c * T_x
        - f_u
    )

    r_v = (
        u * v_x + v * v_y
        + p_y
        - nu * (v_xx + v_yy)
        - lambda_2 * c * T_y
        - beta_T * T
        - beta_c * c
        - f_v
    )

    r_T = (
        u * T_x + v * T_y
        - alpha * (T_xx + T_yy)
        - gamma_1 * (T_x**2 + T_y**2)
        - Q
    )

    r_c = (
        u * c_x + v * c_y
        - D * (c_xx + c_yy)
        - gamma_2 * div_c_grad_T
        - S
    )

    return r_cont, r_u, r_v, r_T, r_c



mse_loss = nn.MSELoss()


def compute_pde_loss(model, batch, cfg):
    r_cont, r_u, r_v, r_T, r_c = pde_residuals(model, batch["x_f"], batch["y_f"], cfg, batch)

    w, r_total = residual_attention_weights(
        r_cont, r_u, r_v, r_T, r_c,
        eta=cfg["attention_eta"],
        eps=cfg["attention_eps"],
        detach=True,
    )

    loss_cont = torch.mean(w * r_cont**2)
    loss_u = torch.mean(w * r_u**2)
    loss_v = torch.mean(w * r_v**2)
    loss_T = torch.mean(w * r_T**2)
    loss_c = torch.mean(w * r_c**2)

    loss_pde = loss_cont + loss_u + loss_v + loss_T + loss_c

    detail = {
        "loss_cont": loss_cont,
        "loss_u": loss_u,
        "loss_v": loss_v,
        "loss_T": loss_T,
        "loss_c": loss_c,
        "attention_mean": torch.mean(w),
        "residual_mean": torch.mean(r_total),
    }
    return loss_pde, detail


def compute_bc_loss(model, batch):
    pred_bc = model(batch["x_bc"], batch["y_bc"])
    target_bc = batch["target_bc"]
    return mse_loss(pred_bc, target_bc)


def compute_data_loss(model, batch):
    pred_train = model(batch["x_train"], batch["y_train"])
    return mse_loss(pred_train, batch["target_train"])


def compute_validation_loss(model, batch):
    model.eval()
    with torch.no_grad():
        pred_val = model(batch["x_val"], batch["y_val"])
        loss_val = mse_loss(pred_val, batch["target_val"])
    model.train()
    return loss_val


def compute_total_loss(model, batch, cfg):
    loss_pde, pde_detail = compute_pde_loss(model, batch, cfg)
    loss_bc = compute_bc_loss(model, batch)
    loss_data = compute_data_loss(model, batch)

    total_loss = (
        cfg["lambda_pde"] * loss_pde
        + cfg["lambda_bc"] * loss_bc
        + cfg["lambda_data"] * loss_data
    )
    return total_loss, loss_pde, loss_bc, loss_data, pde_detail



def compute_metrics_one(pred, true, eps=1e-12):
    diff = pred - true
    mse = np.mean(diff**2)
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



def plot_field(X, Y, Z, title, save_path, cmap="rainbow"):
    plt.figure(figsize=(7, 6))
    plt.pcolormesh(X, Y, Z, shading="auto", cmap=cmap)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_combined_prediction(fields_pred, X, Y, save_path):

    names = ["u", "v", "T", "p", "c"]
    titles = ["u_pred", "v_pred", "T_pred", "p_pred", "c_pred"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (name, title) in enumerate(zip(names, titles)):
        im = axes[i].pcolormesh(X, Y, fields_pred[name], shading="auto", cmap="rainbow")
        axes[i].set_title(title)
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        fig.colorbar(im, ax=axes[i])

    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_curves(log_df, save_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(log_df["epoch"], log_df["total_loss"], label="total_loss")
    plt.plot(log_df["epoch"], log_df["pde_loss"], label="pde_loss")
    plt.plot(log_df["epoch"], log_df["bc_loss"], label="bc_loss")
    plt.plot(log_df["epoch"], log_df["data_loss"], label="data_loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="val_loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300)
    plt.close()



def build_training_batch(cfg, device, p_ref):


    x_f, y_f = sample_interior_points(cfg["N_f"], cfg, device, requires_grad=True)
    src = source_terms_detached(x_f, y_f, cfg, p_ref)


    x_bc, y_bc = sample_boundary_points(cfg["N_bc_each"], cfg, device)
    target_bc = exact_solution_tensor_detached(x_bc, y_bc, p_ref)


    x_data, y_data, target_data = sample_supervised_data(cfg["N_data"], cfg, device, p_ref)
    split = split_train_val(x_data, y_data, target_data, train_ratio=0.7)

    batch = {
        "x_f": x_f,
        "y_f": y_f,
        "f_u": src["f_u"],
        "f_v": src["f_v"],
        "Q": src["Q"],
        "S": src["S"],

        "x_bc": x_bc,
        "y_bc": y_bc,
        "target_bc": target_bc,

        "x_train": split["x_train"],
        "y_train": split["y_train"],
        "target_train": split["target_train"],

        "x_val": split["x_val"],
        "y_val": split["y_val"],
        "target_val": split["target_val"],
    }
    return batch


def train(model, cfg, dirs, p_ref):
    device = cfg["device"]
    batch = build_training_batch(cfg, device, p_ref)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["scheduler_step"],
        gamma=cfg["scheduler_gamma"]
    )

    best_model_path = os.path.join(dirs["models"], "best_model.pth")
    last_model_path = os.path.join(dirs["models"], "last_model.pth")

    log_rows = []
    best_val = float("inf")
    cumulative_time = 0.0

    print(f"Using device: {device}")
    print("Training starts...")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()

        optimizer.zero_grad()
        total_loss, loss_pde, loss_bc, loss_data, pde_detail = compute_total_loss(model, batch, cfg)
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        val_loss = compute_validation_loss(model, batch)

        epoch_time = time.time() - t0
        cumulative_time += epoch_time

        row = {
            "epoch": epoch,
            "total_loss": float(total_loss.item()),
            "pde_loss": float(loss_pde.item()),
            "bc_loss": float(loss_bc.item()),
            "data_loss": float(loss_data.item()),
            "val_loss": float(val_loss.item()),
            "epoch_time": epoch_time,
            "cumulative_time": cumulative_time,
            "attention_mean": float(pde_detail["attention_mean"].item()),
            "residual_mean": float(pde_detail["residual_mean"].item()),
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_rows.append(row)

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            torch.save(model.state_dict(), best_model_path)

        if epoch % cfg["print_every"] == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:5d}/{cfg['epochs']}] | "
                f"total={total_loss.item():.6e} | "
                f"pde={loss_pde.item():.6e} | "
                f"bc={loss_bc.item():.6e} | "
                f"data={loss_data.item():.6e} | "
                f"val={val_loss.item():.6e} | "
                f"time={epoch_time:.3f}s"
            )


    torch.save(model.state_dict(), last_model_path)


    if cfg["use_lbfgs"]:
        print("LBFGS fine-tuning starts...")
        optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=cfg["lbfgs_max_iter"],
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            loss, _, _, _, _ = compute_total_loss(model, batch, cfg)
            loss.backward()
            return loss

        t0 = time.time()
        optimizer_lbfgs.step(closure)
        lbfgs_time = time.time() - t0
        cumulative_time += lbfgs_time

        val_loss = compute_validation_loss(model, batch)
        if val_loss.item() < best_val:
            best_val = val_loss.item()
            torch.save(model.state_dict(), best_model_path)

        torch.save(model.state_dict(), last_model_path)
        print(f"LBFGS finished in {lbfgs_time:.3f}s")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(dirs["logs"], "training_log.csv"), index=False)

    return log_df, best_model_path, last_model_path, cumulative_time



def predict_on_grid(model, cfg, device, p_ref):
    X, Y, x_t, y_t = generate_test_grid(cfg, device)

    model.eval()
    with torch.no_grad():
        pred = model(x_t, y_t).cpu().numpy()


    exact = exact_solution_tensor_detached(x_t, y_t, p_ref).cpu().numpy()

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
    for name in ["u", "v", "T", "p", "c"]:

        plot_field(
            X, Y, fields_pred[name],
            title=f"{name} prediction",
            save_path=os.path.join(dirs["figures"], f"{name}_pred.png"),
            cmap="rainbow",
        )
        save_txt_field(X, Y, fields_pred[name], os.path.join(dirs["txt"], f"{name}_pred.txt"))


        plot_field(
            X, Y, fields_exact[name],
            title=f"{name} exact",
            save_path=os.path.join(dirs["figures"], f"{name}_exact.png"),
            cmap="rainbow",
        )
        save_txt_field(X, Y, fields_exact[name], os.path.join(dirs["txt"], f"{name}_exact.txt"))


        plot_field(
            X, Y, fields_error[name],
            title=f"{name} error",
            save_path=os.path.join(dirs["figures"], f"{name}_error.png"),
            cmap="rainbow",
        )
        save_txt_field(X, Y, fields_error[name], os.path.join(dirs["txt"], f"{name}_error.txt"))


    plot_combined_prediction(
        fields_pred, X, Y,
        os.path.join(dirs["figures"], "combined_prediction.png")
    )


def save_metrics(metrics_df, dirs, device, best_model_path, last_model_path, total_train_time, wall_train_time, eval_time):
    metrics_csv = os.path.join(dirs["logs"], "metrics.csv")
    metrics_txt = os.path.join(dirs["logs"], "metrics.txt")

    metrics_df.to_csv(metrics_csv, index=False)

    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write("Residual-Attention PINN Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"device: {device}\n")
        f.write(f"best_model: {best_model_path}\n")
        f.write(f"last_model: {last_model_path}\n")
        f.write(f"training_time_seconds: {total_train_time:.6f}\n")
        f.write(f"wall_clock_training_seconds: {wall_train_time:.6f}\n")
        f.write(f"evaluation_time_seconds: {eval_time:.6f}\n\n")
        f.write(metrics_df.to_string(index=False))



def main():
    try:
        set_seed(CONFIG["seed"])
        dirs = prepare_dirs(CONFIG)
        device = CONFIG["device"]


        p_ref = get_p_ref(device)


        model = ResidualAttentionPINN(CONFIG).to(device)


        wall_train_start = time.time()
        log_df, best_model_path, last_model_path, total_train_time = train(model, CONFIG, dirs, p_ref)
        wall_train_time = time.time() - wall_train_start


        model.load_state_dict(torch.load(best_model_path, map_location=device))


        eval_start = time.time()
        X, Y, fields_pred, fields_exact, fields_error = predict_on_grid(model, CONFIG, device, p_ref)
        eval_time = time.time() - eval_start


        plot_loss_curves(log_df, dirs["figures"])


        metrics_df = compute_all_metrics(fields_pred, fields_exact)
        save_metrics(
            metrics_df, dirs, device,
            best_model_path, last_model_path,
            total_train_time, wall_train_time, eval_time
        )


        save_all_outputs(X, Y, fields_pred, fields_exact, fields_error, dirs)

        print("\nTraining finished successfully.")
        print(f"Best model saved to: {best_model_path}")
        print(f"Last model saved to: {last_model_path}")
        print(f"Training log saved to: {os.path.join(dirs['logs'], 'training_log.csv')}")
        print(f"Metrics saved to: {os.path.join(dirs['logs'], 'metrics.csv')}")
        print(f"Figures saved to: {dirs['figures']}")
        print(f"TXT fields saved to: {dirs['txt']}")
        print("\nMetrics:")
        print(metrics_df.to_string(index=False))

    except Exception as e:
        print("程序运行发生异常：")
        print(str(e))
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()