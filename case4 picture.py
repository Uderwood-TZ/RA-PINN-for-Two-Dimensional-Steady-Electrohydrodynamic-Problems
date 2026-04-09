import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



CASE_ROOT = Path.cwd()

ALGORITHM_DIRS = {
    "pure PINN": CASE_ROOT / "PINN" / "results" / "txt",
    "LSTM-PINN": CASE_ROOT / "LSTM-PINN" / "results" / "txt",
    "RA-PINN": CASE_ROOT / "剩余注意力PINN" / "results" / "txt",
}

MANUAL_LOSS_FILES = {
    "pure PINN": None,
    "LSTM-PINN": None,
    "RA-PINN": None,
}

FIELD_ORDER = ["p", "u", "v", "phi", "T", "c"]

FIELD_ALIASES = {
    "p": ["p", "pressure"],
    "u": ["u", "velocity_x", "horizontal_velocity"],
    "v": ["v", "velocity_y", "vertical_velocity"],
    "phi": ["phi", "varphi", "electric_potential", "potential"],
    "T": ["T", "t", "temperature"],
    "c": ["c", "concentration"],
}

KIND_ALIASES = {
    "pred": ["pred", "prediction"],
    "true": ["true", "exact", "gt", "reference"],
    "error": ["error", "abs_error", "absolute_error"],
}

ROW_LABELS = {
    "p": "p",
    "u": "u",
    "v": "v",
    "phi": "φ",
    "T": "T",
    "c": "c",
}

OUTPUT_DIR = CASE_ROOT / "comparison_figure_output"
OUTPUT_PNG = OUTPUT_DIR / "case4_f4_style_comparison.png"
OUTPUT_PDF = OUTPUT_DIR / "case4_f4_style_comparison.pdf"

SHOW_FIGURE = True
SAVE_PNG = True
SAVE_PDF = True
PNG_DPI = 500

FIELD_CMAP = "rainbow"
ERROR_CMAP = "turbo"

LOSS_LINE_WIDTH = 1.3



@dataclass
class GridField:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass
class AlgorithmFieldBundle:
    pred: Optional[GridField] = None
    true: Optional[GridField] = None
    error: Optional[GridField] = None



def configure_matplotlib() -> None:
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = PNG_DPI
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]


def normalize_field_key(raw_name: str) -> Optional[str]:
    raw_name = raw_name.strip()
    raw_lower = raw_name.lower()
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if raw_name == alias or raw_lower == alias.lower():
                return canonical
    return None


def normalize_kind_key(raw_name: str) -> Optional[str]:
    raw_name = raw_name.strip()
    raw_lower = raw_name.lower()
    for canonical, aliases in KIND_ALIASES.items():
        for alias in aliases:
            if raw_name == alias or raw_lower == alias.lower():
                return canonical
    return None



def read_xyz_txt(file_path: Path) -> GridField:
    try:
        df = pd.read_csv(
            file_path,
            sep=r"[\s,]+",
            engine="python",
            comment="#",
            skip_blank_lines=True
        )
    except Exception as e:
        raise ValueError(f"读取文件失败：{file_path}\n原始错误：{e}")

    if df.empty:
        raise ValueError(f"文件为空：{file_path}")

    lower_cols = [str(c).strip().lower() for c in df.columns]

    if len(lower_cols) >= 3 and lower_cols[0] == "x" and lower_cols[1] == "y":
        x_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        y_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        z_raw = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy()
    else:
        df = pd.read_csv(
            file_path,
            sep=r"[\s,]+",
            engine="python",
            comment="#",
            skip_blank_lines=True,
            header=None
        )
        if df.shape[1] < 3:
            raise ValueError(f"文件格式错误，要求至少三列 x y value：{file_path}")

        x_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        y_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        z_raw = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy()

    valid_mask = np.isfinite(x_raw) & np.isfinite(y_raw) & np.isfinite(z_raw)
    x_raw = x_raw[valid_mask]
    y_raw = y_raw[valid_mask]
    z_raw = z_raw[valid_mask]

    if len(x_raw) == 0:
        raise ValueError(f"文件中没有有效数值数据：{file_path}")

    x_raw = np.round(x_raw, 12)
    y_raw = np.round(y_raw, 12)

    x_unique = np.unique(x_raw)
    y_unique = np.unique(y_raw)

    x_index = {v: i for i, v in enumerate(x_unique)}
    y_index = {v: i for i, v in enumerate(y_unique)}

    z_grid = np.full((len(y_unique), len(x_unique)), np.nan, dtype=float)

    for x_val, y_val, z_val in zip(x_raw, y_raw, z_raw):
        z_grid[y_index[y_val], x_index[x_val]] = z_val

    x_mesh, y_mesh = np.meshgrid(x_unique, y_unique)
    return GridField(x=x_mesh, y=y_mesh, z=z_grid)



def list_all_possible_data_files(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file()]


def should_ignore_plot_file(file_path: Path) -> bool:
    name = file_path.stem if file_path.suffix else file_path.name
    name = name.strip().lower()

    if name.startswith("loss_"):
        return True

    if name.startswith("r_"):
        return True

    return False


def parse_field_filename(file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    if should_ignore_plot_file(file_path):
        return None, None

    name = file_path.stem if file_path.suffix else file_path.name
    name = name.strip()

    match = re.match(r"^(.+?)_([A-Za-z_]+)$", name)
    if not match:
        return None, None

    raw_field, raw_kind = match.groups()
    field_key = normalize_field_key(raw_field)
    kind_key = normalize_kind_key(raw_kind)
    return field_key, kind_key


def collect_field_files(folder: Path) -> Dict[str, Dict[str, Path]]:
    result: Dict[str, Dict[str, Path]] = {}

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder}")

    for file_path in list_all_possible_data_files(folder):
        field_key, kind_key = parse_field_filename(file_path)
        if field_key is None or kind_key is None:
            continue
        result.setdefault(field_key, {})[kind_key] = file_path

    return result


def scan_all_available_fields(algorithm_dirs: Dict[str, Path]) -> Dict[str, Dict[str, Dict[str, Path]]]:
    collection = {}
    for alg_name, folder in algorithm_dirs.items():
        collection[alg_name] = collect_field_files(folder)
    return collection


def choose_fields_to_plot(collection: Dict[str, Dict[str, Dict[str, Path]]]) -> List[str]:
    field_set = set()
    for alg_data in collection.values():
        field_set.update(alg_data.keys())

    ordered = [f for f in FIELD_ORDER if f in field_set]
    extras = sorted([f for f in field_set if f not in ordered])
    return ordered + extras


def build_field_data(
    collection: Dict[str, Dict[str, Dict[str, Path]]],
    fields: List[str]
) -> Dict[str, Dict[str, AlgorithmFieldBundle]]:
    field_data: Dict[str, Dict[str, AlgorithmFieldBundle]] = {}

    shared_true: Dict[str, GridField] = {}
    for field in fields:
        for alg_name, alg_fields in collection.items():
            true_path = alg_fields.get(field, {}).get("true")
            if true_path is not None:
                shared_true[field] = read_xyz_txt(true_path)
                break

    for field in fields:
        field_data[field] = {}
        for alg_name, alg_fields in collection.items():
            bundle = AlgorithmFieldBundle()
            entry = alg_fields.get(field, {})

            if "pred" in entry:
                bundle.pred = read_xyz_txt(entry["pred"])

            if "true" in entry:
                bundle.true = read_xyz_txt(entry["true"])
            elif field in shared_true:
                bundle.true = shared_true[field]

            if "error" in entry:
                bundle.error = read_xyz_txt(entry["error"])
            elif bundle.pred is not None and bundle.true is not None:
                bundle.error = GridField(
                    x=bundle.pred.x,
                    y=bundle.pred.y,
                    z=np.abs(bundle.pred.z - bundle.true.z),
                )

            field_data[field][alg_name] = bundle

    return field_data



def looks_numeric_dataframe(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False

    numeric_cols = 0
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= max(5, int(0.5 * len(s))):
            numeric_cols += 1
    return numeric_cols >= 1


def try_read_table(file_path: Path) -> Optional[pd.DataFrame]:
    readers = [
        lambda p: pd.read_csv(p, sep=r"\s+", engine="python", comment="#"),
        lambda p: pd.read_csv(p, sep=",", engine="python", comment="#"),
        lambda p: pd.read_csv(p, sep="\t", engine="python", comment="#"),
    ]
    for reader in readers:
        try:
            df = reader(file_path)
            if looks_numeric_dataframe(df):
                return df
        except Exception:
            pass
    return None


def infer_loss_xy_from_dataframe(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    lower_cols = {str(c).strip().lower(): c for c in df.columns}

    x_candidates = ["step", "steps", "epoch", "epochs", "iteration", "iterations"]
    y_candidates = ["total_loss", "loss", "train_loss", "training_loss", "total"]

    x_col = None
    y_col = None

    for key in x_candidates:
        if key in lower_cols:
            x_col = lower_cols[key]
            break
    for key in y_candidates:
        if key in lower_cols:
            y_col = lower_cols[key]
            break

    if x_col is not None and y_col is not None:
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if mask.sum() >= 5:
            return x[mask], y[mask]

    numeric_cols = []
    for col in df.columns:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy()
        if np.isfinite(arr).sum() >= 5:
            numeric_cols.append(arr)

    if len(numeric_cols) == 0:
        return None, None
    if len(numeric_cols) == 1:
        y = numeric_cols[0]
        mask = np.isfinite(y) & (y > 0)
        if mask.sum() >= 5:
            x = np.arange(1, mask.sum() + 1, dtype=float)
            return x, y[mask]
        return None, None

    x = numeric_cols[0]
    y = numeric_cols[1]
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if mask.sum() >= 5:
        return x[mask], y[mask]

    return None, None


def infer_loss_xy_from_log_text(file_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None

    x_values = []
    y_values = []

    patterns = [
        re.compile(
            r"Epoch\s*\[\s*(\d+)\s*/\s*\d+\s*\].*?Total\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
            re.IGNORECASE
        ),
        re.compile(
            r"step\s*[:=]\s*(\d+).*?(?:total_loss|loss|train_loss|training_loss|total)\s*[:=]\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
            re.IGNORECASE
        ),
        re.compile(
            r"epoch\s*[:=]\s*(\d+).*?(?:total_loss|loss|train_loss|training_loss|total)\s*[:=]\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
            re.IGNORECASE
        ),
    ]

    for pattern in patterns:
        matches = pattern.findall(text)
        if len(matches) >= 5:
            for x_str, y_str in matches:
                try:
                    x_val = float(x_str)
                    y_val = float(y_str)
                    if np.isfinite(x_val) and np.isfinite(y_val) and y_val > 0:
                        x_values.append(x_val)
                        y_values.append(y_val)
                except Exception:
                    continue
            if len(x_values) >= 5:
                return np.array(x_values, dtype=float), np.array(y_values, dtype=float)

    return None, None


def try_get_loss_xy(file_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    df = try_read_table(file_path)
    if df is not None:
        x, y = infer_loss_xy_from_dataframe(df)
        if x is not None and y is not None:
            return x, y

    x, y = infer_loss_xy_from_log_text(file_path)
    if x is not None and y is not None:
        return x, y

    return None, None


def find_loss_file(algorithm_txt_dir: Path, manual_loss_file: Optional[Path]) -> Optional[Path]:
    if manual_loss_file is not None:
        manual_loss_file = Path(manual_loss_file)
        if manual_loss_file.exists():
            return manual_loss_file

    algorithm_root = algorithm_txt_dir.parent.parent if algorithm_txt_dir.name == "txt" else algorithm_txt_dir

    candidate_dirs = [
        algorithm_root,
        algorithm_root / "results",
        algorithm_root / "results" / "logs",
        algorithm_root / "results" / "txt",
    ]

    keywords = ["loss", "history", "train", "training", "curve", "log"]

    candidates = []
    for base in candidate_dirs:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(k in p.name.lower() for k in keywords):
                candidates.append(p)

    def sort_key(path: Path):
        name = path.name.lower()
        return (
            0 if "loss" in name else 1,
            0 if "history" in name else 1,
            len(name),
        )

    candidates = sorted(set(candidates), key=sort_key)

    for file_path in candidates:
        x, y = try_get_loss_xy(file_path)
        if x is not None and y is not None and len(x) >= 5:
            print(f"[损失文件] {file_path}")
            return file_path

    return None



def get_global_pred_limits(true_grid: Optional[GridField], pred_grids: List[Optional[GridField]]) -> Tuple[float, float]:
    arrays = []
    if true_grid is not None:
        arrays.append(true_grid.z)
    for g in pred_grids:
        if g is not None:
            arrays.append(g.z)

    if not arrays:
        return 0.0, 1.0

    vmin = min(np.nanmin(a) for a in arrays)
    vmax = max(np.nanmax(a) for a in arrays)

    if np.isclose(vmin, vmax):
        eps = 1e-12 if abs(vmin) < 1 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps

    return float(vmin), float(vmax)


def get_global_error_limits(error_grids: List[Optional[GridField]]) -> Tuple[float, float]:
    arrays = [g.z for g in error_grids if g is not None]
    if not arrays:
        return 0.0, 1.0

    vmin = 0.0
    vmax = max(float(np.nanmax(a)) for a in arrays)
    if np.isclose(vmax, 0.0):
        vmax = 1e-12
    return vmin, vmax


def set_left_row_label(ax, field_key: str) -> None:
    label = ROW_LABELS.get(field_key, field_key)
    ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=15, va="center", ha="right")
    ax.yaxis.set_label_coords(-0.09, 0.5)


def style_axis(
    ax,
    field_key: str = "",
    show_left_label: bool = False,
    show_xlabel: bool = False,
    show_xticklabels: bool = False,
    show_yticklabels: bool = False
) -> None:
    ax.tick_params(labelsize=6, length=2)

    if show_left_label:
        set_left_row_label(ax, field_key)
    else:
        ax.set_ylabel("")

    if show_xlabel:
        ax.set_xlabel("x", fontsize=7)
    else:
        ax.set_xlabel("")

    ax.tick_params(
        bottom=True,
        left=True,
        labelbottom=show_xticklabels,
        labelleft=show_yticklabels
    )



def plot_loss_panel(fig: plt.Figure, spec) -> None:
    ax = fig.add_subplot(spec)
    plotted = False
    cached_curves = []

    for alg_name, txt_dir in ALGORITHM_DIRS.items():
        loss_file = find_loss_file(txt_dir, MANUAL_LOSS_FILES.get(alg_name))
        if loss_file is None:
            print(f"[警告] 未找到损失文件：{alg_name}")
            continue

        x, y = try_get_loss_xy(loss_file)
        if x is None or y is None:
            print(f"[警告] 无法解析损失文件：{loss_file}")
            continue

        ax.plot(x, y, linewidth=LOSS_LINE_WIDTH, label=alg_name)
        cached_curves.append((x, y))
        plotted = True

    ax.set_title("Case 4", fontsize=9, pad=3)
    ax.set_xlabel("step", fontsize=7)
    ax.set_ylabel("total loss", fontsize=7)
    ax.tick_params(labelsize=6, length=2)
    ax.grid(True, which="both", alpha=0.25)

    if not plotted:
        ax.text(
            0.5, 0.5,
            "No readable loss history",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=9
        )
        return

    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=6, frameon=True)

    inset = inset_axes(ax, width="28%", height="42%", loc="upper right", borderpad=0.9)
    for x, y in cached_curves:
        if len(x) < 10:
            continue
        start = int(len(x) * 0.75)
        inset.plot(x[start:], y[start:], linewidth=1.0)

    inset.set_yscale("log")
    inset.tick_params(labelsize=5, length=1.5)
    inset.grid(True, which="both", alpha=0.2)

    try:
        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.4", lw=0.8)
    except Exception:
        pass



def plot_comparison_figure(
    field_data: Dict[str, Dict[str, AlgorithmFieldBundle]],
    fields: List[str],
    algorithm_order: List[str]
) -> plt.Figure:
    if len(fields) == 0:
        raise RuntimeError("没有检测到可绘制字段，请确认存在类似 c_true、c_pred、c_abs_error 这类文件。")

    num_fields = len(fields)
    fig = plt.figure(figsize=(18.2, 2.28 * num_fields + 3.08))

    gs = gridspec.GridSpec(
        nrows=num_fields + 1,
        ncols=10,
        figure=fig,
        height_ratios=[1.10] + [1.50] * num_fields,
        width_ratios=[1, 1, 1, 1, 0.048, 0.13, 1, 1, 1, 0.048],
        hspace=0.30,
        wspace=0.020,
    )

    fig.suptitle("Case 4: F4-style comparison", fontsize=13, y=0.985)

    plot_loss_panel(fig, gs[0, :])

    for row_idx, field in enumerate(fields, start=1):
        bundles = field_data[field]
        is_bottom_row = (row_idx == num_fields)

        reference_true = None
        for alg_name in algorithm_order:
            if bundles[alg_name].true is not None:
                reference_true = bundles[alg_name].true
                break

        pred_grids = [bundles[alg].pred for alg in algorithm_order]
        err_grids = [bundles[alg].error for alg in algorithm_order]

        pred_vmin, pred_vmax = get_global_pred_limits(reference_true, pred_grids)
        err_vmin, err_vmax = get_global_error_limits(err_grids)

        ax_exact = fig.add_subplot(gs[row_idx, 0])
        ax_pure = fig.add_subplot(gs[row_idx, 1])
        ax_lstm = fig.add_subplot(gs[row_idx, 2])
        ax_ra = fig.add_subplot(gs[row_idx, 3])
        cax_pred = fig.add_subplot(gs[row_idx, 4])
        ax_gap = fig.add_subplot(gs[row_idx, 5])
        ax_err_pure = fig.add_subplot(gs[row_idx, 6])
        ax_err_lstm = fig.add_subplot(gs[row_idx, 7])
        ax_err_ra = fig.add_subplot(gs[row_idx, 8])
        cax_err = fig.add_subplot(gs[row_idx, 9])

        ax_gap.axis("off")

        pred_mesh = None
        err_mesh = None

        if reference_true is not None:
            pred_mesh = ax_exact.pcolormesh(
                reference_true.x, reference_true.y, reference_true.z,
                shading="auto", cmap=FIELD_CMAP,
                vmin=pred_vmin, vmax=pred_vmax
            )
            ax_exact.set_aspect("equal")
        else:
            ax_exact.text(0.5, 0.5, "Missing Data", ha="center", va="center", transform=ax_exact.transAxes)

        style_axis(
            ax_exact,
            field_key=field,
            show_left_label=True,
            show_xlabel=is_bottom_row,
            show_xticklabels=is_bottom_row,
            show_yticklabels=True
        )

        for ax, alg_name in zip([ax_pure, ax_lstm, ax_ra], algorithm_order):
            grid = bundles[alg_name].pred
            if grid is not None:
                pred_mesh = ax.pcolormesh(
                    grid.x, grid.y, grid.z,
                    shading="auto", cmap=FIELD_CMAP,
                    vmin=pred_vmin, vmax=pred_vmax
                )
                ax.set_aspect("equal")
            else:
                ax.text(0.5, 0.5, "Missing Data", ha="center", va="center", transform=ax.transAxes)

            style_axis(
                ax,
                show_left_label=False,
                show_xlabel=is_bottom_row,
                show_xticklabels=is_bottom_row,
                show_yticklabels=False
            )

        for ax, alg_name in zip([ax_err_pure, ax_err_lstm, ax_err_ra], algorithm_order):
            grid = bundles[alg_name].error
            if grid is not None:
                err_mesh = ax.pcolormesh(
                    grid.x, grid.y, grid.z,
                    shading="auto", cmap=ERROR_CMAP,
                    vmin=err_vmin, vmax=err_vmax
                )
                ax.set_aspect("equal")
            else:
                ax.text(0.5, 0.5, "Missing Data", ha="center", va="center", transform=ax.transAxes)

            style_axis(
                ax,
                show_left_label=False,
                show_xlabel=is_bottom_row,
                show_xticklabels=is_bottom_row,
                show_yticklabels=False
            )

        if row_idx == 1:
            ax_exact.set_title("Exact", fontsize=8, pad=2.5)
            ax_pure.set_title("pure PINN", fontsize=8, pad=2.5)
            ax_lstm.set_title("LSTM-PINN", fontsize=8, pad=2.5)
            ax_ra.set_title("RA-PINN", fontsize=8, pad=2.5)
            ax_err_pure.set_title("pure error", fontsize=8, pad=2.5)
            ax_err_lstm.set_title("LSTM error", fontsize=8, pad=2.5)
            ax_err_ra.set_title("RA error", fontsize=8, pad=2.5)

        if pred_mesh is not None:
            cb1 = fig.colorbar(pred_mesh, cax=cax_pred)
            cb1.ax.tick_params(labelsize=4.8, pad=0.6, length=1.2)
            cb1.ax.yaxis.set_ticks_position("right")
            cb1.ax.yaxis.set_label_position("right")
            cb1.ax.tick_params(labelleft=False, labelright=True)
        else:
            cax_pred.axis("off")

        if err_mesh is not None:
            cb2 = fig.colorbar(err_mesh, cax=cax_err)
            cb2.ax.tick_params(labelsize=4.8, pad=0.6, length=1.2)
            cb2.ax.yaxis.set_ticks_position("right")
            cb2.ax.yaxis.set_label_position("right")
            cb2.ax.tick_params(labelleft=False, labelright=True)
        else:
            cax_err.axis("off")

    fig.subplots_adjust(left=0.055, right=0.986, top=0.963, bottom=0.055)
    return fig



def validate_directories() -> None:
    missing = []
    for name, path in ALGORITHM_DIRS.items():
        if not path.exists():
            missing.append(f"{name}: {path}")

    if missing:
        msg = "\n".join(missing)
        raise FileNotFoundError(
            "以下算法数据文件夹不存在，请确认当前工作目录是否为算例根目录：\n" + msg
        )


def save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_PNG:
        fig.savefig(OUTPUT_PNG, dpi=PNG_DPI, bbox_inches="tight")
        print(f"[已保存] {OUTPUT_PNG}")

    if SAVE_PDF:
        fig.savefig(OUTPUT_PDF, bbox_inches="tight")
        print(f"[已保存] {OUTPUT_PDF}")


def main() -> None:
    configure_matplotlib()
    validate_directories()

    collection = scan_all_available_fields(ALGORITHM_DIRS)
    fields = choose_fields_to_plot(collection)
    algorithm_order = list(ALGORITHM_DIRS.keys())
    field_data = build_field_data(collection, fields)

    print("=" * 100)
    print("当前工作目录：", CASE_ROOT)
    print("检测到的字段：", ", ".join(fields) if fields else "未检测到")
    print("输出目录：", OUTPUT_DIR)
    print("=" * 100)

    fig = plot_comparison_figure(field_data, fields, algorithm_order)
    save_figure(fig)

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()