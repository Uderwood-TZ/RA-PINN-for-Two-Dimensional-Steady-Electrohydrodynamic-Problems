

import os
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



CONFIG = {
    "seed": 42,
    "x_min": 0.0,
    "x_max": 1.0,
    "y_min": 0.0,
    "y_max": 1.0,


    "nu": 0.01,
    "kappa": 0.02,
    "D": 0.015,
    "beta_tc": 0.5,
    "gamma_ct": 0.3,


    "Fe_x": 0.2,
    "Fe_y": -0.1,

    "N_bc_each": 500,
    "N_f": 10000,
    "N_data": 5000,


    "N_test_x": 200,
    "N_test_y": 200,


    "input_dim": 2,
    "output_dim": 5,
    "hidden_dim": 128,
    "num_hidden_layers": 6,
    "activation": "tanh",


    "adam_epochs": 5000,
    "lbfgs_enable": True,
    "lbfgs_max_iter": 500,
    "lr_adam": 1e-3,
    "lr_lbfgs": 1.0,


    "w_pde": 1.0,
    "w_bc": 20.0,
    "w_data": 10.0,


    "print_every": 100,
    "save_dir": "results_striped_electrode_pinn",
}



def make_dirs(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "txt"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)


make_dirs(CONFIG["save_dir"])



PI = math.pi


def u_exact(x, y):
    return torch.sin(PI * x) * torch.cos(PI * y) + 0.1 * torch.sin(2 * PI * x) * torch.cos(3 * PI * y)


def v_exact(x, y):
    return torch.cos(PI * x) * torch.sin(PI * y) + 0.1 * torch.cos(2 * PI * x) * torch.sin(3 * PI * y)


def T_exact(x, y):
    return torch.sin(PI * x) * torch.sin(PI * y) + 0.2 * torch.cos(PI * x) * torch.cos(PI * y)


def p_exact(x, y):
    return torch.cos(PI * x) * torch.cos(PI * y) + 0.2 * torch.sin(2 * PI * x) * torch.sin(2 * PI * y)


def c_exact(x, y):
    return torch.cos(PI * x) * torch.sin(PI * y) + 0.15 * torch.cos(2 * PI * x) * torch.cos(3 * PI * y)



def gradients(outputs, inputs):

    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]


def second_gradients(outputs, inputs):

    first = gradients(outputs, inputs)
    second = gradients(first, inputs)
    return second



def exact_fields_and_derivatives(x, y):

    u = u_exact(x, y)
    v = v_exact(x, y)
    T = T_exact(x, y)
    p = p_exact(x, y)
    c = c_exact(x, y)

    u_x = gradients(u, x)
    u_y = gradients(u, y)
    u_xx = second_gradients(u, x)
    u_yy = second_gradients(u, y)

    v_x = gradients(v, x)
    v_y = gradients(v, y)
    v_xx = second_gradients(v, x)
    v_yy = second_gradients(v, y)

    T_x = gradients(T, x)
    T_y = gradients(T, y)
    T_xx = second_gradients(T, x)
    T_yy = second_gradients(T, y)

    p_x = gradients(p, x)
    p_y = gradients(p, y)

    c_x = gradients(c, x)
    c_y = gradients(c, y)
    c_xx = second_gradients(c, x)
    c_yy = second_gradients(c, y)

    return {
        "u": u, "v": v, "T": T, "p": p, "c": c,
        "u_x": u_x, "u_y": u_y, "u_xx": u_xx, "u_yy": u_yy,
        "v_x": v_x, "v_y": v_y, "v_xx": v_xx, "v_yy": v_yy,
        "T_x": T_x, "T_y": T_y, "T_xx": T_xx, "T_yy": T_yy,
        "p_x": p_x, "p_y": p_y,
        "c_x": c_x, "c_y": c_y, "c_xx": c_xx, "c_yy": c_yy
    }



def manufactured_source_terms(x, y, cfg):
    nu = cfg["nu"]
    kappa = cfg["kappa"]
    D = cfg["D"]
    beta_tc = cfg["beta_tc"]
    gamma_ct = cfg["gamma_ct"]
    Fe_x = cfg["Fe_x"]
    Fe_y = cfg["Fe_y"]

    fd = exact_fields_and_derivatives(x, y)

    u = fd["u"]
    v = fd["v"]
    T = fd["T"]
    p = fd["p"]
    c = fd["c"]

    u_x = fd["u_x"]
    u_y = fd["u_y"]
    u_xx = fd["u_xx"]
    u_yy = fd["u_yy"]

    v_x = fd["v_x"]
    v_y = fd["v_y"]
    v_xx = fd["v_xx"]
    v_yy = fd["v_yy"]

    T_x = fd["T_x"]
    T_y = fd["T_y"]
    T_xx = fd["T_xx"]
    T_yy = fd["T_yy"]

    p_x = fd["p_x"]
    p_y = fd["p_y"]

    c_x = fd["c_x"]
    c_y = fd["c_y"]
    c_xx = fd["c_xx"]
    c_yy = fd["c_yy"]


    f_mass = u_x + v_y



    f_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy) - Fe_x


    f_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy) - Fe_y


    f_T = u * T_x + v * T_y - kappa * (T_xx + T_yy) - beta_tc * u * c


    f_c = u * c_x + v * c_y - D * (c_xx + c_yy) - gamma_ct * T

    return {
        "f_mass": f_mass,
        "f_u": f_u,
        "f_v": f_v,
        "f_T": f_T,
        "f_c": f_c
    }



def sample_interior_points(n, cfg, requires_grad=False):
    x = torch.rand(n, 1) * (cfg["x_max"] - cfg["x_min"]) + cfg["x_min"]
    y = torch.rand(n, 1) * (cfg["y_max"] - cfg["y_min"]) + cfg["y_min"]
    x = x.to(device)
    y = y.to(device)

    if requires_grad:
        x.requires_grad_(True)
        y.requires_grad_(True)
    return x, y


def sample_boundary_points_each_side(n_each, cfg):

    x_min, x_max = cfg["x_min"], cfg["x_max"]
    y_min, y_max = cfg["y_min"], cfg["y_max"]


    xb = torch.rand(n_each, 1) * (x_max - x_min) + x_min
    yb = torch.full((n_each, 1), y_min)


    xt = torch.rand(n_each, 1) * (x_max - x_min) + x_min
    yt = torch.full((n_each, 1), y_max)


    xl = torch.full((n_each, 1), x_min)
    yl = torch.rand(n_each, 1) * (y_max - y_min) + y_min


    xr = torch.full((n_each, 1), x_max)
    yr = torch.rand(n_each, 1) * (y_max - y_min) + y_min

    x = torch.cat([xb, xt, xl, xr], dim=0).to(device)
    y = torch.cat([yb, yt, yl, yr], dim=0).to(device)
    return x, y


def sample_supervised_data(n, cfg):

    x, y = sample_interior_points(n, cfg, requires_grad=False)
    with torch.no_grad():
        u = u_exact(x, y)
        v = v_exact(x, y)
        T = T_exact(x, y)
        p = p_exact(x, y)
        c = c_exact(x, y)
        target = torch.cat([u, v, T, p, c], dim=1)

    return x, y, target


def train_val_split(x, y, target, train_ratio=0.7):
    n = x.shape[0]
    idx = torch.randperm(n, device=device)
    n_train = int(train_ratio * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    data = {
        "x_train": x[train_idx],
        "y_train": y[train_idx],
        "target_train": target[train_idx],
        "x_val": x[val_idx],
        "y_val": y[val_idx],
        "target_val": target[val_idx],
    }
    return data


def generate_test_grid(cfg):
    nx = cfg["N_test_x"]
    ny = cfg["N_test_y"]

    x = np.linspace(cfg["x_min"], cfg["x_max"], nx)
    y = np.linspace(cfg["y_min"], cfg["y_max"], ny)
    X, Y = np.meshgrid(x, y)

    XY = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    x_t = torch.tensor(XY[:, 0:1], dtype=torch.float32, device=device)
    y_t = torch.tensor(XY[:, 1:2], dtype=torch.float32, device=device)

    return X, Y, x_t, y_t



class FCNN_PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=5, hidden_dim=128, num_hidden_layers=6, activation="tanh"):
        super().__init__()

        if activation == "tanh":
            act = nn.Tanh
        elif activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            raise ValueError("Unsupported activation")

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)  # [N,2]
        out = self.net(inp)             # [N,5]
        return out



mse_loss = nn.MSELoss()


def pde_residuals(model, x, y, cfg):

    out = model(x, y)
    u = out[:, 0:1]
    v = out[:, 1:2]
    T = out[:, 2:3]
    p = out[:, 3:4]
    c = out[:, 4:5]

    u_x = gradients(u, x)
    u_y = gradients(u, y)
    u_xx = second_gradients(u, x)
    u_yy = second_gradients(u, y)

    v_x = gradients(v, x)
    v_y = gradients(v, y)
    v_xx = second_gradients(v, x)
    v_yy = second_gradients(v, y)

    T_x = gradients(T, x)
    T_y = gradients(T, y)
    T_xx = second_gradients(T, x)
    T_yy = second_gradients(T, y)

    p_x = gradients(p, x)
    p_y = gradients(p, y)

    c_x = gradients(c, x)
    c_y = gradients(c, y)
    c_xx = second_gradients(c, x)
    c_yy = second_gradients(c, y)

    sources = manufactured_source_terms(x, y, cfg)

    nu = cfg["nu"]
    kappa = cfg["kappa"]
    D = cfg["D"]
    beta_tc = cfg["beta_tc"]
    gamma_ct = cfg["gamma_ct"]
    Fe_x = cfg["Fe_x"]
    Fe_y = cfg["Fe_y"]

    r_mass = u_x + v_y - sources["f_mass"]

    r_u = (
        u * u_x + v * u_y
        + p_x
        - nu * (u_xx + u_yy)
        - Fe_x
        - sources["f_u"]
    )

    r_v = (
        u * v_x + v * v_y
        + p_y
        - nu * (v_xx + v_yy)
        - Fe_y
        - sources["f_v"]
    )

    r_T = (
        u * T_x + v * T_y
        - kappa * (T_xx + T_yy)
        - beta_tc * u * c
        - sources["f_T"]
    )

    r_c = (
        u * c_x + v * c_y
        - D * (c_xx + c_yy)
        - gamma_ct * T
        - sources["f_c"]
    )

    return r_mass, r_u, r_v, r_T, r_c


def compute_pde_loss(model, x_f, y_f, cfg):
    r_mass, r_u, r_v, r_T, r_c = pde_residuals(model, x_f, y_f, cfg)
    loss = (
        torch.mean(r_mass ** 2)
        + torch.mean(r_u ** 2)
        + torch.mean(r_v ** 2)
        + torch.mean(r_T ** 2)
        + torch.mean(r_c ** 2)
    )
    return loss


def compute_bc_loss(model, x_bc, y_bc):
    pred = model(x_bc, y_bc)
    with torch.no_grad():
        target = torch.cat([
            u_exact(x_bc, y_bc),
            v_exact(x_bc, y_bc),
            T_exact(x_bc, y_bc),
            p_exact(x_bc, y_bc),
            c_exact(x_bc, y_bc),
        ], dim=1)

    return mse_loss(pred, target)


def compute_data_loss(model, x_d, y_d, target_d):
    pred = model(x_d, y_d)
    return mse_loss(pred, target_d)


def compute_val_loss(model, x_val, y_val, target_val):
    model.eval()
    with torch.no_grad():
        pred = model(x_val, y_val)
        loss = mse_loss(pred, target_val)
    return loss



def train_model(model, cfg):
    save_dir = cfg["save_dir"]
    model_dir = os.path.join(save_dir, "models")
    best_model_path = os.path.join(model_dir, "best_model.pt")


    x_f, y_f = sample_interior_points(cfg["N_f"], cfg, requires_grad=True)
    x_bc, y_bc = sample_boundary_points_each_side(cfg["N_bc_each"], cfg)
    x_data, y_data, target_data = sample_supervised_data(cfg["N_data"], cfg)
    split_data = train_val_split(x_data, y_data, target_data, train_ratio=0.7)

    x_train = split_data["x_train"]
    y_train = split_data["y_train"]
    target_train = split_data["target_train"]
    x_val = split_data["x_val"]
    y_val = split_data["y_val"]
    target_val = split_data["target_val"]


    optimizer_adam = optim.Adam(model.parameters(), lr=cfg["lr_adam"])
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=cfg["lr_lbfgs"],
        max_iter=cfg["lbfgs_max_iter"],
        max_eval=cfg["lbfgs_max_iter"],
        history_size=50,
        tolerance_grad=1e-9,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )

    w_pde = cfg["w_pde"]
    w_bc = cfg["w_bc"]
    w_data = cfg["w_data"]

    history = []
    best_val = float("inf")


    print("\n========== Adam Training ==========")
    t0_adam = time.time()

    for epoch in range(1, cfg["adam_epochs"] + 1):
        model.train()
        optimizer_adam.zero_grad()

        loss_pde = compute_pde_loss(model, x_f, y_f, cfg)
        loss_bc = compute_bc_loss(model, x_bc, y_bc)
        loss_data = compute_data_loss(model, x_train, y_train, target_train)

        loss = w_pde * loss_pde + w_bc * loss_bc + w_data * loss_data
        loss.backward()
        optimizer_adam.step()

        val_loss = compute_val_loss(model, x_val, y_val, target_val)

        record = {
            "epoch": epoch,
            "stage": "Adam",
            "total_loss": float(loss.item()),
            "pde_loss": float(loss_pde.item()),
            "bc_loss": float(loss_bc.item()),
            "data_loss": float(loss_data.item()),
            "val_loss": float(val_loss.item()),
        }
        history.append(record)

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            torch.save(model.state_dict(), best_model_path)

        if epoch % cfg["print_every"] == 0 or epoch == 1:
            print(
                f"[Adam] Epoch {epoch:6d}/{cfg['adam_epochs']} | "
                f"total={loss.item():.6e} | "
                f"pde={loss_pde.item():.6e} | "
                f"bc={loss_bc.item():.6e} | "
                f"data={loss_data.item():.6e} | "
                f"val={val_loss.item():.6e}"
            )

    adam_time = time.time() - t0_adam


    lbfgs_time = 0.0
    if cfg["lbfgs_enable"]:
        print("\n========== LBFGS Fine-tuning ==========")
        t0_lbfgs = time.time()

        lbfgs_iter = [0]

        def closure():
            optimizer_lbfgs.zero_grad()
            model.train()

            loss_pde = compute_pde_loss(model, x_f, y_f, cfg)
            loss_bc = compute_bc_loss(model, x_bc, y_bc)
            loss_data = compute_data_loss(model, x_train, y_train, target_train)
            loss_total = w_pde * loss_pde + w_bc * loss_bc + w_data * loss_data

            loss_total.backward()

            lbfgs_iter[0] += 1
            if lbfgs_iter[0] % 10 == 0 or lbfgs_iter[0] == 1:
                val_loss = compute_val_loss(model, x_val, y_val, target_val)
                print(
                    f"[LBFGS] Iter {lbfgs_iter[0]:4d}/{cfg['lbfgs_max_iter']} | "
                    f"total={loss_total.item():.6e} | "
                    f"pde={loss_pde.item():.6e} | "
                    f"bc={loss_bc.item():.6e} | "
                    f"data={loss_data.item():.6e} | "
                    f"val={val_loss.item():.6e}"
                )
            return loss_total

        optimizer_lbfgs.step(closure)


        with torch.enable_grad():
            loss_pde = compute_pde_loss(model, x_f, y_f, cfg)
            loss_bc = compute_bc_loss(model, x_bc, y_bc)
            loss_data = compute_data_loss(model, x_train, y_train, target_train)
            loss = w_pde * loss_pde + w_bc * loss_bc + w_data * loss_data
            val_loss = compute_val_loss(model, x_val, y_val, target_val)

        record = {
            "epoch": cfg["adam_epochs"] + 1,
            "stage": "LBFGS",
            "total_loss": float(loss.item()),
            "pde_loss": float(loss_pde.item()),
            "bc_loss": float(loss_bc.item()),
            "data_loss": float(loss_data.item()),
            "val_loss": float(val_loss.item()),
        }
        history.append(record)

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            torch.save(model.state_dict(), best_model_path)

        lbfgs_time = time.time() - t0_lbfgs


    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, "train_log.csv"), index=False)


    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\nBest validation loss = {best_val:.6e}")
    print(f"Best model saved to: {best_model_path}")

    return model, history_df, adam_time, lbfgs_time



def compute_metrics(pred, exact):
    pred = pred.reshape(-1)
    exact = exact.reshape(-1)

    diff = pred - exact

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    l2_rel = np.linalg.norm(diff, 2) / (np.linalg.norm(exact, 2) + 1e-12)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "L2_relative": l2_rel
    }


def test_model(model, cfg):
    save_dir = cfg["save_dir"]

    X, Y, x_t, y_t = generate_test_grid(cfg)

    t0_test = time.time()
    model.eval()
    with torch.no_grad():
        pred = model(x_t, y_t).cpu().numpy()

        u_pred = pred[:, 0].reshape(cfg["N_test_y"], cfg["N_test_x"])
        v_pred = pred[:, 1].reshape(cfg["N_test_y"], cfg["N_test_x"])
        T_pred = pred[:, 2].reshape(cfg["N_test_y"], cfg["N_test_x"])
        p_pred = pred[:, 3].reshape(cfg["N_test_y"], cfg["N_test_x"])
        c_pred = pred[:, 4].reshape(cfg["N_test_y"], cfg["N_test_x"])

        u_ex = u_exact(x_t, y_t).cpu().numpy().reshape(cfg["N_test_y"], cfg["N_test_x"])
        v_ex = v_exact(x_t, y_t).cpu().numpy().reshape(cfg["N_test_y"], cfg["N_test_x"])
        T_ex = T_exact(x_t, y_t).cpu().numpy().reshape(cfg["N_test_y"], cfg["N_test_x"])
        p_ex = p_exact(x_t, y_t).cpu().numpy().reshape(cfg["N_test_y"], cfg["N_test_x"])
        c_ex = c_exact(x_t, y_t).cpu().numpy().reshape(cfg["N_test_y"], cfg["N_test_x"])

    test_time = time.time() - t0_test

    results = {
        "u": {"pred": u_pred, "exact": u_ex, "error": np.abs(u_pred - u_ex)},
        "v": {"pred": v_pred, "exact": v_ex, "error": np.abs(v_pred - v_ex)},
        "T": {"pred": T_pred, "exact": T_ex, "error": np.abs(T_pred - T_ex)},
        "p": {"pred": p_pred, "exact": p_ex, "error": np.abs(p_pred - p_ex)},
        "c": {"pred": c_pred, "exact": c_ex, "error": np.abs(c_pred - c_ex)},
    }

    metrics_all = {}
    for var in ["u", "v", "T", "p", "c"]:
        metrics_all[var] = compute_metrics(results[var]["pred"], results[var]["exact"])


    print("\n========== Test Metrics ==========")
    for var in ["u", "v", "T", "p", "c"]:
        m = metrics_all[var]
        print(
            f"{var:>2s} | "
            f"MSE={m['MSE']:.6e}, "
            f"RMSE={m['RMSE']:.6e}, "
            f"MAE={m['MAE']:.6e}, "
            f"L2={m['L2_relative']:.6e}"
        )


    metrics_rows = []
    with open(os.path.join(save_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Variable, MSE, RMSE, MAE, L2_relative\n")
        for var in ["u", "v", "T", "p", "c"]:
            m = metrics_all[var]
            line = f"{var}, {m['MSE']:.10e}, {m['RMSE']:.10e}, {m['MAE']:.10e}, {m['L2_relative']:.10e}\n"
            f.write(line)
            metrics_rows.append({
                "variable": var,
                "MSE": m["MSE"],
                "RMSE": m["RMSE"],
                "MAE": m["MAE"],
                "L2_relative": m["L2_relative"],
            })

    pd.DataFrame(metrics_rows).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

    return X, Y, results, metrics_all, test_time



def export_field_txt(X, Y, Z, file_path):
    data = np.column_stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])
    np.savetxt(file_path, data, fmt="%.10e", header="x y value", comments="")



def plot_field(X, Y, Z, title, save_path):
    plt.figure(figsize=(7, 6))
    plt.pcolormesh(X, Y, Z, shading="auto", cmap="rainbow")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_all_field_plots_and_txt(X, Y, results, cfg):
    fig_dir = os.path.join(cfg["save_dir"], "figures")
    txt_dir = os.path.join(cfg["save_dir"], "txt")

    for var in ["u", "v", "T", "p", "c"]:
        for kind in ["pred", "exact", "error"]:
            Z = results[var][kind]

            png_path = os.path.join(fig_dir, f"{var}_{kind}.png")
            txt_path = os.path.join(txt_dir, f"{var}_{kind}.txt")

            plot_field(X, Y, Z, f"{var} {kind}", png_path)
            export_field_txt(X, Y, Z, txt_path)



def plot_loss_curve(history_df, cfg):
    fig_dir = os.path.join(cfg["save_dir"], "figures")


    epochs = history_df["epoch"].values

    curves = [
        ("total_loss", "Total Loss", "total_loss_curve.png"),
        ("pde_loss", "PDE Loss", "pde_loss_curve.png"),
        ("bc_loss", "BC Loss", "bc_loss_curve.png"),
        ("data_loss", "Data Loss", "data_loss_curve.png"),
        ("val_loss", "Validation Loss", "val_loss_curve.png"),
    ]

    for col, title, filename in curves:
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history_df[col].values, linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(title)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, filename), dpi=300)
        plt.close()



def save_time_log(cfg, total_train_time, adam_time, lbfgs_time, test_time, total_run_time):
    file_path = os.path.join(cfg["save_dir"], "time_log.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Total training time : {total_train_time:.6f} s\n")
        f.write(f"Adam training time  : {adam_time:.6f} s\n")
        f.write(f"LBFGS training time : {lbfgs_time:.6f} s\n")
        f.write(f"Total test time     : {test_time:.6f} s\n")
        f.write(f"Total run time      : {total_run_time:.6f} s\n")

    print("\n========== Time Log ==========")
    print(f"Total training time : {total_train_time:.6f} s")
    print(f"Adam training time  : {adam_time:.6f} s")
    print(f"LBFGS training time : {lbfgs_time:.6f} s")
    print(f"Total test time     : {test_time:.6f} s")
    print(f"Total run time      : {total_run_time:.6f} s")



def main():
    total_start = time.time()


    model = FCNN_PINN(
        input_dim=CONFIG["input_dim"],
        output_dim=CONFIG["output_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_hidden_layers=CONFIG["num_hidden_layers"],
        activation=CONFIG["activation"]
    ).to(device)


    train_start = time.time()
    model, history_df, adam_time, lbfgs_time = train_model(model, CONFIG)
    total_train_time = time.time() - train_start


    X, Y, results, metrics_all, test_time = test_model(model, CONFIG)


    save_all_field_plots_and_txt(X, Y, results, CONFIG)


    plot_loss_curve(history_df, CONFIG)


    total_run_time = time.time() - total_start
    save_time_log(CONFIG, total_train_time, adam_time, lbfgs_time, test_time, total_run_time)

    print(f"\nAll results saved in: {CONFIG['save_dir']}")


if __name__ == "__main__":
    main()