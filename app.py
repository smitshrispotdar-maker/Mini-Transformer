import math, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Shared with MiniGPT_Colab_Training.ipynb: identical architecture and tokenizer.
from minigpt_colab import (
    MiniGPT,
    config_fingerprint,
    count_parameters,
    evaluate_token_stream,
    history_from_csv,
    load_colab_checkpoint,
    seed_everything,
)

st.set_page_config(page_title="Mini-Transformer (nanoGPT)", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

st.html("""
<style>
:root {
  --canvas: #FAF9F6;
  --surface: #FFFFFF;
  --border: #E7E5E4;
  --text: #1C1917;
  --muted: #57534E;
  --accent: #B45309;
  --slate: #1C1917;
  --amber: #9A3412;
  --stone: #A8A29E;
  --soft: #F5F5F4;
  --sans: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  --mono: "JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}
.stApp { background: var(--canvas); color: var(--text); font-family: var(--sans); }
.block-container { max-width: 1480px; padding: 1.25rem 2.25rem 3rem 2.25rem; }
[data-testid="stHeader"] { background: rgba(250, 249, 246, 0); }
[data-testid="stSidebar"], [data-testid="stToolbar"] { display: none; }
h1, h2, h3, h4, h5, h6 { color: var(--text); font-family: var(--sans); }
.stMarkdown p, .stCaption { color: var(--muted); font-size: 12px; }
.app-header { text-align: center; padding: 8px 0 18px; border-bottom: 1px solid var(--border); margin-bottom: 18px; }
.app-title { font-size: 26px; font-weight: 650; letter-spacing: -0.03em; color: var(--text); line-height: 1.2; }
.config-summary { display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--soft); font-family: var(--mono); font-size: 11px; color: var(--slate); }
.config-summary span { white-space: nowrap; }
.config-summary b { color: var(--muted); font-family: var(--sans); font-weight: 600; }
.corpus-strip { margin: 12px 0 14px; padding: 9px 12px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--muted); font-size: 12px; }
.corpus-strip b { color: var(--text); font-family: var(--mono); font-weight: 600; }
div[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid var(--border) !important; border-radius: 8px !important; background: var(--surface) !important; box-shadow: none !important; padding: 16px !important; }
.card-title { font-size: 14px; font-weight: 650; letter-spacing: -0.01em; color: var(--text); }
.card-subtitle { margin-top: 3px; font-size: 11px; color: var(--muted); }
.kpi-card { min-height: 106px; padding: 14px 16px; border: 1px solid var(--border); border-radius: 8px; background: var(--surface); }
.kpi-label { text-transform: uppercase; letter-spacing: 0.08em; font-size: 10px; font-weight: 650; color: var(--muted); }
.kpi-value { margin-top: 10px; font-family: var(--mono); font-size: 23px; letter-spacing: -0.025em; color: var(--text); }
.kpi-sub { margin-top: 7px; font-size: 11px; color: var(--muted); }
[data-testid="stTabs"] [role="tablist"] { gap: 6px; border-bottom: 1px solid var(--border); }
[data-testid="stTabs"] [role="tab"] { padding: 10px 14px; border-radius: 6px 6px 0 0; color: var(--muted); font-size: 13px; font-weight: 600; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: var(--surface); color: var(--accent); border-bottom: 2px solid var(--accent); }
.stButton button { border-radius: 8px !important; font-family: var(--sans) !important; font-size: 13px !important; font-weight: 600 !important; }
.stButton button[kind="primary"] { background: var(--slate) !important; color: #FFFFFF !important; border: 1px solid var(--slate) !important; }
.stButton button[kind="primary"]:hover { background: #292524 !important; border-color: #292524 !important; }
.stButton button[kind="secondary"] { background: var(--surface) !important; color: var(--slate) !important; border: 1px solid var(--border) !important; }
.stButton button[kind="secondary"]:hover { background: var(--soft) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 8px !important; border-width: 1px !important; }
[data-testid="stAlert"] div { font-size: 12px !important; }
.kv-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.kv-table th { padding: 8px 6px; text-align: left; text-transform: uppercase; letter-spacing: 0.06em; font-size: 9px; color: var(--muted); font-weight: 650; }
.kv-table td { padding: 9px 6px; border-top: 1px solid var(--border); color: var(--text); }
.kv-table td.kv-value { font-family: var(--mono); font-weight: 600; }
.kv-table td.kv-detail { text-align: right; color: var(--muted); font-size: 11px; }
.footnote { margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--muted); font-size: 11px; }
@media (max-width: 900px) { .block-container { padding: 1rem 1rem 2rem 1rem; } }
</style>
""")

def ui_card(title, subtitle=None):
    container = st.container(border=True)
    with container:
        st.html(f'<div class="card-title">{title}</div>')
        if subtitle:
            st.html(f'<div class="card-subtitle">{subtitle}</div>')
    return container

def ui_kpi(label, value, subtext):
    st.html(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{subtext}</div></div>'
    )

def ui_axis(ax, xlabel=None, ylabel=None, grid_axis="y"):
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis=grid_axis, color="#E7E5E4", linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D6D3D1")
    ax.tick_params(colors="#57534E", labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color="#57534E", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color="#57534E", fontsize=10)

def ui_loss_chart(history, figsize=(7.6, 3.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor("#FFFFFF")
    ax.plot(history["step"], history["train"], color="#B45309", linewidth=1.8, marker="o", markersize=3, label="Train")
    ax.plot(history["step"], history["val"], color="#78716C", linewidth=1.8, marker="o", markersize=3, label="Validation")
    ui_axis(ax, "Training iteration", "Cross-entropy loss", "both")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def ui_sensitivity_chart(data, metric, figsize=(7.2, 3.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor("#FFFFFF")
    bars = ax.bar(data["Model"], data[metric], color="#B45309", width=0.58)
    if metric == "Accuracy (%)":
        label_fmt = "%.2f"
    elif metric == "Parameters":
        label_fmt = "%.0f"
    elif metric == "Train time (s)":
        label_fmt = "%.2f"
    elif metric == "Perplexity":
        label_fmt = "%.3f"
    else:
        label_fmt = "%.4f"
    ax.bar_label(bars, fmt=label_fmt, padding=3, fontsize=8, color="#57534E")
    ui_axis(ax, None, metric, "y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", color="#57534E", fontsize=8)
    ax.margins(y=0.18)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def ui_confusion_preview(metrics_data, itos, top_n, figsize=(7.2, 3.4), cmap="Oranges"):
    true, pred = metrics_data["y_true"], metrics_data["y_pred"]
    counts = np.bincount(true, minlength=len(itos))
    top_ids = np.argsort(counts)[::-1][:top_n]
    mapping = {int(v): i for i, v in enumerate(top_ids)}
    other_idx = len(top_ids)
    yt = np.array([mapping.get(int(v), other_idx) for v in true])
    yp = np.array([mapping.get(int(v), other_idx) for v in pred])
    labels = [repr(itos[int(i)]) for i in top_ids] + ["other"]
    cm = confusion_matrix(yt, yp, labels=list(range(len(labels))), normalize="true")
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor("#FFFFFF")
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap=cmap, ax=ax, annot=False, cbar_kws={"shrink": 0.8}, linewidths=0.4, linecolor="#FFFFFF")
    ax.set_xlabel("Predicted character", color="#57534E", fontsize=10)
    ax.set_ylabel("Actual character", color="#57534E", fontsize=10)
    ax.tick_params(labelsize=8, colors="#57534E")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def ui_config_table(result, chars):
    cfg = result["config"]
    rows = [
        ("Token embedding", f"{len(chars)} × {cfg['n_embd']}", "vocab × dim"),
        ("Position embedding", f"{cfg['block_size']} × {cfg['n_embd']}", "context × dim"),
        ("Transformer blocks", f"{cfg['n_layer']} × attention + FFN", f"{cfg['n_head']} heads · {cfg['n_embd'] // cfg['n_head']} per head"),
        ("LM linear head", f"{cfg['n_embd']} × {len(chars)}", "dim → vocab"),
        ("Total params", f"{result['params']:,}", "trainable"),
        ("Hardware", result["device"], "active device"),
        ("Train / val split", "90% / 10%", "character indices"),
    ]
    body = "".join(
        f"<tr><td>{name}</td><td class='kv-value'>{value}</td><td class='kv-detail'>{detail}</td></tr>"
        for name, value, detail in rows
    )
    st.html(f'<table class="kv-table"><thead><tr><th>Component</th><th>Value</th><th>Detail</th></tr></thead><tbody>{body}</tbody></table>')

DEFAULT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SEED = 1337

@st.cache_data
def load_corpus(uploaded_bytes=None):
    if uploaded_bytes is not None:
        return uploaded_bytes.decode("utf-8", errors="replace")
    path = Path("input.txt")
    if not path.exists():
        try:
            urllib.request.urlretrieve(DEFAULT_URL, path)
        except Exception as exc:
            raise RuntimeError("Could not download Tiny Shakespeare. Upload the input.txt dataset in the sidebar.") from exc
    return path.read_text(encoding="utf-8")

def make_batch(source, batch_size, block_size, device):
    ix = torch.randint(len(source)-block_size-1, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def eval_loss(model, source, batch_size, block_size, device, eval_iters=20):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x,y = make_batch(source, batch_size, block_size, device)
        _, loss = model(x,y); losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

def train_model(text, n_embd, n_head, n_layer, block_size, dropout, batch_size, max_iters, lr, eval_interval, seed=SEED):
    seed_everything(seed)
    chars = sorted(set(text)); stoi = {c:i for i,c in enumerate(chars)}; itos = {i:c for c,i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(.90*len(data)); train_data, val_data = data[:split], data[split:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(len(chars), n_embd, n_head, n_layer, block_size, dropout).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = {"step": [], "train": [], "val": []}
    start = time.time()
    for step in range(max_iters+1):
        if step % eval_interval == 0 or step == max_iters:
            hist["step"].append(step)
            hist["train"].append(eval_loss(model, train_data, batch_size, block_size, device))
            hist["val"].append(eval_loss(model, val_data, batch_size, block_size, device))
        if step == max_iters: break
        x,y = make_batch(train_data, batch_size, block_size, device)
        _, loss = model(x,y)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    elapsed = time.time()-start
    result = {"model": model, "stoi": stoi, "itos": itos, "chars": chars, "history": hist,
              "params": params, "seconds": elapsed, "device": device, "config": dict(n_embd=n_embd,n_head=n_head,n_layer=n_layer,block_size=block_size,dropout=dropout,batch_size=batch_size,max_iters=max_iters,lr=lr),
              "train_data": train_data, "val_data": val_data}
    return result

def test_metrics(result, max_tokens=30000):
    model, device = result["model"], result["device"]
    chars = result["chars"]
    block = result["config"]["block_size"]

    # Colab/BPE checkpoints: recompute locally if validation.txt was exported;
    # otherwise report the metrics recorded by the notebook (never invented).
    if result.get("tokenizer") is not None:
        meta = result.get("metadata") or {}
        recorded = meta.get("evaluation") or {}
        checkpoint_dir = Path(result.get("checkpoint_dir", ""))
        validation_file = checkpoint_dir / "validation.txt"
        if validation_file.is_file():
            tokenizer = result["tokenizer"]
            data = torch.tensor(tokenizer.encode(validation_file.read_text(encoding="utf-8"), add_special=False), dtype=torch.long)
            if len(data) > block:
                values = evaluate_token_stream(model, data, block, device, max_windows=512)
                values["chars"] = chars
                values["token_based"] = True
                return values
        loss = float(recorded.get("test_loss", recorded.get("val_loss", float("nan"))))
        perplexity = float(recorded.get("test_perplexity", recorded.get("val_perplexity", float("nan"))))
        if math.isnan(perplexity) and not math.isnan(loss):
            perplexity = float(math.exp(min(loss, 20)))
        empty = np.array([], dtype=np.int64)
        return {
            "loss": loss, "perplexity": perplexity,
            "accuracy": float(recorded.get("accuracy", 0.0)),
            "y_true": empty, "y_pred": empty, "chars": chars, "token_based": True,
        }

    # Original character-level path.
    data = result["val_data"][:max_tokens+1]
    if len(data) <= block: return {}
    X = torch.stack([data[i:i+block] for i in range(0, min(len(data)-block, 4096), block)])
    Y = torch.stack([data[i+1:i+block+1] for i in range(0, min(len(data)-block, 4096), block)])
    model.eval()
    with torch.no_grad():
        logits, loss = model(X.to(device), Y.to(device))
        pred = logits.argmax(-1).cpu().reshape(-1).numpy()
        true = Y.reshape(-1).numpy()
    acc = float((pred == true).mean())
    return {"loss": float(loss.item()), "perplexity": float(math.exp(min(loss.item(), 20))),
            "accuracy": acc, "y_true": true, "y_pred": pred, "chars": chars}

if "result" not in st.session_state: st.session_state.result = None
if "comparisons" not in st.session_state: st.session_state.comparisons = []

st.session_state.setdefault("_run_training_now", False)
torch.set_num_threads(1)

def _queue_training():
    st.session_state["_run_training_now"] = True

def _load_colab_checkpoint(path_text, device="cpu"):
    """Load a Colab artifact folder into the same result shape used by the UI."""
    path = Path(path_text).expanduser()
    model, tokenizer, config, metadata = load_colab_checkpoint(path, device=device)
    history = history_from_csv(path / "training_metrics.csv")
    st.session_state.result = {
        "model": model,
        "tokenizer": tokenizer,
        "stoi": {token: i for i, token in enumerate(tokenizer.vocab)},
        "itos": {i: token for i, token in enumerate(tokenizer.vocab)},
        "chars": tokenizer.vocab,
        "history": history,
        "params": count_parameters(model),
        "seconds": float((metadata.get("training") or {}).get("duration_seconds", 0.0)),
        "device": str(next(model.parameters()).device),
        "config": config,
        "train_data": None,
        "val_data": None,
        "metadata": metadata,
        "checkpoint_dir": str(path.resolve()),
        "source": "colab",
    }
    st.session_state.comparisons = []
    st.session_state["_run_training_now"] = False
    return st.session_state.result

def active_model_info(result):
    """Read-only description of the exact model instance behind the UI."""
    cfg = result["config"]
    model = result["model"]
    architecture = (
        f"{cfg['n_layer']} layers · {cfg['n_head']} heads · {cfg['n_embd']} embed · "
        f"block {cfg['block_size']} · dropout {cfg.get('dropout', 0)}"
    )
    tokenizer = result.get("tokenizer")
    if tokenizer is not None:
        meta = result.get("metadata") or {}
        dataset_cfg = ((meta.get("dataset") or {}).get("config") or {})
        sources = [
            f"{name} ({settings.get('max_docs')} docs)"
            for name, settings in dataset_cfg.items() if settings.get("enabled")
        ]
        dataset = ", ".join(sources) if sources else "Colab-exported corpus (see metadata.json)"
        return {
            "Model name": "Colab-exported MiniGPT (byte-level BPE)",
            "Tokenizer type": tokenizer.config.get("type", "byte_level_bpe"),
            "Vocabulary size": f"{tokenizer.vocab_size:,} tokens",
            "Parameter count": f"{count_parameters(model):,}",
            "Architecture": architecture,
            "Dataset": dataset,
            "Checkpoint path": result.get("checkpoint_dir", "unknown"),
        }
    return {
        "Model name": "Legacy character-level MiniGPT (trained in this session)",
        "Tokenizer type": "Character-level stoi/itos (from input text)",
        "Vocabulary size": f"{len(result['chars']):,} characters",
        "Parameter count": f"{count_parameters(model):,}",
        "Architecture": architecture,
        "Dataset": result.get("dataset", "input.txt (Tiny Shakespeare character corpus)"),
        "Checkpoint path": result.get("checkpoint_path", "None - no checkpoint file loaded"),
    }

st.html("""
<style>
.status-pill { display: inline-flex; align-items: center; padding: 5px 10px; border: 1px solid #E7E5E4; border-radius: 999px; font-size: 11px; font-weight: 650; }
.status-pill-ready { background: #F5F5F4; color: #57534E; }
.status-pill-trained { background: #FFF7ED; color: #9A3412; border-color: #FED7AA; }
.blueprint-row { padding: 10px 0; border-top: 1px solid #E7E5E4; }
.blueprint-row:first-child { padding-top: 2px; border-top: 0; }
.blueprint-head { display: flex; justify-content: space-between; gap: 16px; color: #1C1917; font-size: 12px; font-weight: 650; }
.blueprint-count { font-family: var(--mono); font-size: 11px; color: #57534E; white-space: nowrap; }
.blueprint-detail { margin-top: 3px; color: #57534E; font-size: 10px; }
.blueprint-track { height: 6px; margin-top: 7px; border-radius: 999px; background: #F5F5F4; overflow: hidden; }
.blueprint-fill { height: 100%; border-radius: 999px; background: #B45309; }
.blueprint-fill-muted { background: #A8A29E; }
.partition-bar { display: flex; height: 10px; border: 1px solid #E7E5E4; border-radius: 999px; background: #FFFFFF; overflow: hidden; }
.partition-train { background: #B45309; }
.partition-val { background: #A8A29E; }
.partition-labels { display: flex; justify-content: space-between; gap: 12px; margin-top: 8px; color: #57534E; font-size: 11px; }
.partition-labels b { color: #1C1917; font-family: var(--mono); }
.target-list { list-style: none; padding: 0; margin: 12px 0 14px; }
.target-list li { display: flex; justify-content: space-between; gap: 16px; padding: 7px 0; border-top: 1px solid #E7E5E4; color: #57534E; font-size: 11px; }
.target-list li b { color: #1C1917; font-family: var(--mono); }
.dashboard-empty { padding: 22px 18px; border: 1px solid #E7E5E4; border-radius: 8px; background: #FFFFFF; color: #57534E; font-size: 12px; text-align: center; }
</style>
""")

def ui_parameter_breakdown(vocab_size, n_embd, n_head, n_layer, block_size):
    head_size = n_embd // n_head
    attention = n_head * 3 * n_embd * head_size + (n_head * head_size) * n_embd + n_embd
    feed_forward = 8 * n_embd * n_embd + 5 * n_embd
    blocks = n_layer * (attention + feed_forward + 4 * n_embd)
    head_group = n_embd * vocab_size + vocab_size + 2 * n_embd
    token_embedding = vocab_size * n_embd
    positional = block_size * n_embd
    total = token_embedding + positional + blocks + head_group
    return total, [
        ("Token Embedding", f"{vocab_size:,} × {n_embd} symbols", token_embedding),
        ("Positional Encoding", f"{block_size} × {n_embd} context slots", positional),
        (f"Transformer Blocks × {n_layer}", "self-attention + feed-forward + LayerNorm", blocks),
        ("LM Linear Head", f"{n_embd} × {vocab_size:,} + final LayerNorm", head_group),
    ]

def ui_blueprint_rows(rows):
    total = sum(count for _, _, count in rows)
    for index, (label, detail, count) in enumerate(rows):
        share = 100 * count / total if total else 0.0
        fill_class = "blueprint-fill" if index == 2 else "blueprint-fill blueprint-fill-muted"
        st.html(
            f'<div class="blueprint-row"><div class="blueprint-head"><span>{label}</span>'
            f'<span class="blueprint-count">{count:,} · {share:.1f}%</span></div>'
            f'<div class="blueprint-detail">{detail}</div>'
            f'<div class="blueprint-track"><div class="{fill_class}" style="width:{share:.2f}%"></div></div></div>'
        )

def ui_empty_tab(title):
    with ui_card(title, "No trained model in this session"):
        st.markdown("Click **Train & Evaluate Model** in the **Model Hyperparameters** drawer to populate this view. No metrics are shown before a real run.")

st.html("""
<div class="app-header">
  <div class="app-title">Mini-Transformer (nanoGPT)</div>
</div>
""")

summary_slot = st.empty()
with st.expander("Model Hyperparameters", expanded=True):
    st.caption("Dataset source, architecture, and training controls")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        emb = st.slider("Embedding dimension", 32, 256, 48, 16, key="train_emb")
    with h2:
        heads = st.select_slider("Attention heads", options=[2,4,6,8], value=4, key="train_heads")
    with h3:
        layers = st.slider("Transformer layers", 1, 6, 2, key="train_layers")
    with h4:
        block = st.select_slider("Context length", options=[32,64,96,128,192,256], value=32, key="train_block")
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        dropout = st.slider("Dropout", 0.0, 0.5, 0.10, 0.05, key="train_dropout")
    with i2:
        batch = st.select_slider("Batch size", options=[8,16,32,64], value=16, key="train_batch")
    with i3:
        iterations = st.slider("Training iterations", 50, 2000, 80, 10, key="train_iters")
    with i4:
        lr = st.select_slider("Learning rate", options=[0.0001,0.0003,0.0005,0.001,0.002], value=0.002, format_func=lambda x:f"{x:g}", key="train_lr")
    j1, j2, j3, j4 = st.columns(4)
    with j1:
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=1337)
    with j2:
        upload = st.file_uploader("Optional: upload Tiny Shakespeare input.txt", type=["txt"])
        st.caption("If no file is uploaded, the app uses the local corpus or attempts the standard download.")
    with j3:
        st.button("Train & Evaluate Model", type="primary", on_click=_queue_training, key="btn_train_drawer")
    with j4:
        reset_button = st.button("Reset", type="secondary", key="reset_cta")

    st.markdown("**Colab checkpoint (optional)**")
    ck1, ck2 = st.columns([4, 1])
    with ck1:
        ckpt_path = st.text_input(
            "Folder with model.pt, config.json, tokenizer.json",
            value="mini_gpt_trained_model", key="ckpt_path",
        )
    with ck2:
        load_ckpt = st.button("Load checkpoint", type="secondary", key="load_colab_cta")
    st.caption(
        f"Loading activates the exported model for this session on "
        f"{'CUDA' if torch.cuda.is_available() else 'CPU'}. "
        f"Shared architecture fingerprint: {config_fingerprint()}. "
        "It never overwrites the legacy character model, input.txt, or existing checkpoints."
    )
    if load_ckpt:
        try:
            _load_colab_checkpoint(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu")
            st.success(f"Loaded Colab checkpoint from {ckpt_path}")
            st.rerun()
        except Exception as exc:
            st.error(f"Checkpoint load failed: {exc}")

summary_slot.html(
    '<div class="config-summary">'
    f'<span><b>Embedding Dim</b> {emb}</span>'
    f'<span><b>Attention Heads</b> {heads}</span>'
    f'<span><b>Layers</b> {layers}</span>'
    f'<span><b>Block Size</b> {block}</span>'
    f'<span><b>Dropout</b> {dropout:.2f}</span>'
    f'<span><b>Batch Size</b> {batch}</span>'
    f'<span><b>Learning Rate</b> {lr:g}</span>'
    f'<span><b>Iterations</b> {iterations}</span>'
    '</div>'
)

if reset_button:
    st.session_state.result = None
    st.session_state.comparisons = []
    st.session_state["_run_training_now"] = False
    st.rerun()

try:
    corpus = load_corpus(upload.getvalue() if upload else None)
except Exception as e:
    st.error(str(e)); st.stop()

training_slot = st.container()
with training_slot:
    if st.session_state.get("_run_training_now"):
        st.session_state["_run_training_now"] = False
        if emb % heads != 0:
            st.error("Embedding dimension must be divisible by the number of attention heads.")
        else:
            with st.status("Training Mini-Transformer...", expanded=True) as status:
                try:
                    st.session_state.result = None
                    st.session_state.comparisons = []
                    trained = train_model(corpus, emb, heads, layers, block, dropout, batch, iterations, lr, eval_interval=20, seed=int(seed))
                    trained["source"] = "legacy"
                    trained["dataset"] = upload.name if upload else "input.txt (Tiny Shakespeare character corpus)"
                    trained["checkpoint_path"] = "None - trained in memory in this Streamlit session"
                    st.session_state.result = trained
                    status.update(label="Training complete", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Training failed", state="error", expanded=True)
                    st.exception(e)
            if st.session_state.result is not None:
                st.rerun()

chars = sorted(set(corpus))
r = st.session_state.result
metrics = test_metrics(r) if r is not None else None

tab_dash, tab_hist, tab_gen, tab_sens, tab_conf = st.tabs([
    "Dashboard", "Training History", "Text Generation", "Parameter Sensitivity", "Confusion Matrix"
])

with tab_dash:
    active_cfg = r["config"] if r is not None else dict(
        n_embd=emb, n_head=heads, n_layer=layers, block_size=block, dropout=dropout
    )
    a_emb, a_heads = int(active_cfg["n_embd"]), int(active_cfg["n_head"])
    a_layers, a_block = int(active_cfg["n_layer"]), int(active_cfg["block_size"])
    active_chars = r["chars"] if r is not None else chars
    vocab_size = len(active_chars)
    device = r["device"] if r is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    total_params, breakdown = ui_parameter_breakdown(vocab_size, a_emb, a_heads, a_layers, a_block)
    meta = (r.get("metadata") or {}) if r is not None else {}
    dataset_meta = meta.get("dataset") or {}
    is_colab = r is not None and r.get("source") == "colab"
    if is_colab and dataset_meta:
        corpus_value = dataset_meta.get("total_tokens", dataset_meta.get("total_characters", 0))
        corpus_label = f"{int(corpus_value):,}"
        corpus_sub = "tokens from Colab dataset"
    else:
        corpus_label = f"{len(corpus):,}"
        corpus_sub = "characters from input text"
    token_sub = "byte-level BPE vocabulary" if is_colab else "unique character dictionary"
    split_sub = "document-level 90/5/5 split" if is_colab else "90% train / 10% validation split"
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        ui_kpi("Corpus Size", corpus_label, corpus_sub)
    with k2:
        ui_kpi("Vocabulary Size", f"{vocab_size:,} tokens", token_sub)
    with k3:
        ui_kpi("Model Parameters", f"{total_params:,}", f"{a_layers}L • {a_heads}H • {a_emb}D")
    with k4:
        ui_kpi("Compute & Split", device.upper(), split_sub)

    arch_col, ready_col = st.columns([1.15, 1], gap="medium")
    with arch_col:
        with ui_card("Model Architecture Blueprint", "Live parameter allocation from current controls"):
            ui_blueprint_rows(breakdown)
    with ready_col:
        with ui_card("Dataset Partition & Evaluation Readiness", "Split and active training targets"):
            pill_text = "Colab checkpoint loaded" if is_colab else ("Trained &amp; Evaluated" if r is not None else "Ready to Train")
            pill_class = "status-pill-trained" if r is not None else "status-pill-ready"
            st.html(f'<div class="status-pill {pill_class}">{pill_text}</div>')
            if is_colab and dataset_meta:
                train_tokens = int(dataset_meta.get("train_tokens", 0))
                val_tokens = int(dataset_meta.get("val_tokens", 0))
                test_tokens = int(dataset_meta.get("test_tokens", 0))
                total = max(train_tokens + val_tokens + test_tokens, 1)
                st.html(
                    '<div class="partition-bar">'
                    f'<div class="partition-train" style="width:{100*train_tokens/total:.1f}%"></div>'
                    f'<div class="partition-val" style="width:{100*(val_tokens+test_tokens)/total:.1f}%"></div></div>'
                    f'<div class="partition-labels"><span><b>Train</b> {train_tokens:,} tokens</span>'
                    f'<span><b>Val + test</b> {val_tokens+test_tokens:,} tokens</span></div>'
                )
                active_batch = int(active_cfg.get("batch_size", batch))
                active_lr = float(active_cfg.get("learning_rate", lr))
                active_iters = int(active_cfg.get("max_iters", iterations))
            else:
                train_chars = int(0.9 * len(corpus))
                val_chars = len(corpus) - train_chars
                st.html(
                    '<div class="partition-bar"><div class="partition-train" style="width:90%"></div>'
                    '<div class="partition-val" style="width:10%"></div></div>'
                    f'<div class="partition-labels"><span><b>90% Train</b> {train_chars:,} chars</span>'
                    f'<span><b>10% Validation</b> {val_chars:,} chars</span></div>'
                )
                active_batch, active_lr, active_iters = batch, lr, iterations
            st.html(
                '<ul class="target-list">'
                f'<li><span>Context window</span><b>{a_block}</b></li>'
                f'<li><span>Batch size</span><b>{active_batch}</b></li>'
                f'<li><span>Learning rate</span><b>{active_lr:g}</b></li>'
                f'<li><span>Iterations</span><b>{active_iters}</b></li>'
                '</ul>'
            )

    freq_title = "Colab Dataset Source Attribution" if is_colab else "Top Corpus Character Frequency Distribution"
    freq_subtitle = "Reference input.txt only; training used the exported Colab corpus" if is_colab else "Relative share of the 12 most common symbols"
    with ui_card(freq_title, freq_subtitle):
        counts = pd.Series(list(corpus)).value_counts().head(12)
        labels = ["space" if ch == " " else "\\n" if ch == "\n" else ch for ch in counts.index]
        percents = (counts / len(corpus) * 100).to_numpy()
        fig, ax = plt.subplots(figsize=(10.8, 3.0), dpi=110)
        fig.patch.set_facecolor("#FFFFFF")
        bars = ax.barh(labels[::-1], percents[::-1], color="#B45309", height=0.62)
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in percents[::-1]], padding=4, fontsize=8, color="#57534E")
        ui_axis(ax, "Frequency (%)", None, "x")
        plt.setp(ax.get_yticklabels(), color="#57534E", fontsize=9)
        ax.margins(x=0.18)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_hist:
    if r is None:
        ui_empty_tab("Training History")
    else:
        unit = "token" if metrics.get("token_based") else "character"
        source_note = "Recorded in Colab metadata" if len(metrics["y_true"]) == 0 else "Recomputed locally on exported validation split"
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            ui_kpi("Held-out Loss", f"{metrics['loss']:.4f}", source_note)
        with m2:
            ui_kpi("Perplexity", f"{metrics['perplexity']:.3f}", "exp(cross-entropy)")
        with m3:
            ui_kpi(f"Next-{unit} Accuracy", f"{metrics['accuracy']*100:.2f}%", "Argmax match")
        with m4:
            ui_kpi(f"Evaluated {unit.title()}s", f"{len(metrics['y_true']):,}", "Deterministic held-out subset")

        if not r["history"]["step"]:
            st.info("No training_metrics.csv was found in this checkpoint folder, so loss curves are unavailable. All other metrics above come from the exported metadata.")
        else:
            with ui_card("Loss Trajectory", "Training and validation curves for this run"):
                ui_loss_chart(r["history"], figsize=(9.2, 3.2))
            with ui_card("Checkpoint Logs", "Recorded at each evaluation step"):
                hist_df = pd.DataFrame(r["history"])
                st.dataframe(hist_df, hide_index=True)
                st.download_button("Download training history CSV", hist_df.to_csv(index=False).encode(), "training_history.csv", "text/csv")
        st.caption("Perplexity = exp(cross-entropy loss). Accuracy is exact next-token/character argmax accuracy, not a measure of overall literary quality.")

with tab_gen:
    if r is None:
        ui_empty_tab("Text Generation")
    else:
        is_bpe = r.get("tokenizer") is not None
        default_prompt = ("Explain artificial intelligence in simple English."
                          if is_bpe else "ROMEO:")
        known_defaults = {"", "ROMEO:", "Explain artificial intelligence in simple English."}
        if "gen_prompt" not in st.session_state:
            st.session_state["gen_prompt"] = default_prompt
        elif st.session_state["gen_prompt"] in known_defaults:
            st.session_state["gen_prompt"] = default_prompt

        with ui_card("Active Model", "Exact model used by Generate text"):
            info = active_model_info(r)
            st.markdown("\n".join(f"- **{key}:** {value}" for key, value in info.items()))

        with ui_card("Sampling Controls", "Prompt and decoding parameters"):
            prompt = st.text_area(
                "Prompt", key="gen_prompt", height=100,
                placeholder="Type any prompt, for example: Explain how a computer network works.",
            )
            st.caption("Editable prompt. Your text is preserved across reruns unless you load a different model type.")
            g1, g2, g3 = st.columns(3)
            with g1:
                if is_bpe:
                    max_new = st.slider("New tokens", 20, 600, 160, 20, key="bpe_max_new")
                else:
                    max_new = st.slider("New characters", 50, 1000, 300, 50, key="char_max_new")
            with g2:
                temp = st.slider("Temperature", 0.1, 1.5, 0.8, 0.1, key="gen_temp")
            with g3:
                if is_bpe:
                    top_k = st.slider("Top-k (0 = disabled)", 0, min(100, len(r["chars"])), 40, key="bpe_top_k")
                else:
                    top_k = st.slider("Top-k (0 = disabled)", 0, min(30, len(r["chars"])), 0, key="char_top_k")
            if is_bpe:
                g4, g5, g6 = st.columns(3)
                with g4:
                    top_p = st.slider("Top-p / nucleus", 0.0, 1.0, 0.9, 0.05, key="bpe_top_p")
                with g5:
                    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.15, 0.05, key="bpe_rep_penalty")
                with g6:
                    no_repeat = st.slider("No-repeat n-gram (0 = disabled)", 0, 8, 4, key="bpe_no_repeat")

        with ui_card("Generated Output", "Continuation sampled from the active model"):
            if st.button("Generate text", type="primary", key="generate_cta"):
                prompt = prompt.strip() or ("Explain " if is_bpe else " ")
                st.caption(f"Original prompt: {prompt!r}")
                model = r["model"]
                model.eval()
                if is_bpe:
                    tokenizer = r["tokenizer"]
                    block_size = int(r["config"]["block_size"])
                    full_length = len(tokenizer.encode(prompt, add_special=False)) + 1
                    prompt_ids = tokenizer.encode_prompt(prompt, max_context_tokens=block_size)
                    if full_length > len(prompt_ids):
                        st.warning(
                            f"Prompt was {full_length} tokens but the context window is {block_size}. "
                            f"Kept the last {len(prompt_ids)} tokens."
                        )
                    inp = torch.tensor([prompt_ids], dtype=torch.long, device=r["device"])
                    with torch.inference_mode():
                        generated = model.generate(
                            inp, max_new_tokens=max_new, temperature=temp, top_k=top_k,
                            top_p=top_p, repetition_penalty=repetition_penalty,
                            no_repeat_ngram_size=no_repeat,
                            eos_id=tokenizer.eos_id, stop_at_eos=True,
                        )[0].tolist()
                    new_ids = generated[len(prompt_ids):]
                    continuation = tokenizer.decode(new_ids, skip_special=True, stop_at_eos=True)
                    eos_info = " (EOS reached early)" if tokenizer.eos_id in new_ids else ""
                    st.caption(
                        f"{len(prompt_ids)} prompt tokens → {len(new_ids)} newly generated tokens{eos_info}"
                    )
                else:
                    unknown = [ch for ch in prompt if ch not in r["stoi"]]
                    if unknown:
                        st.warning(f"Prompt includes characters outside the model vocabulary: {unknown!r}. They will be skipped.")
                    ids = [r["stoi"][ch] for ch in prompt if ch in r["stoi"]]
                    if not ids: ids = [0]
                    inp = torch.tensor([ids], dtype=torch.long, device=r["device"])
                    with torch.inference_mode():
                        generated = model.generate(
                            inp, max_new_tokens=max_new, temperature=temp, top_k=top_k
                        )[0].tolist()
                    new_ids = generated[len(ids):]
                    continuation = "".join(r["itos"][i] for i in new_ids)
                    st.caption(f"{len(ids)} prompt characters → {len(new_ids)} newly generated characters")

                if not new_ids:
                    st.warning("The model produced no new tokens. Try a higher temperature or a different prompt.")
                elif not continuation.strip():
                    st.warning(
                        "New tokens were generated but decoded to empty or special-only text "
                        f"(token ids: {new_ids[:10]}). Try a different prompt or temperature."
                    )
                else:
                    st.text_area("Generated continuation", continuation, height=300, key="generated_continuation")
                    st.download_button("Download generated continuation", continuation, "generated_continuation.txt", "text/plain")
            else:
                st.caption("Edit the prompt above and click Generate text to sample from the active model.")

with tab_sens:
    with ui_card("Experiment Runner", "Train smaller and larger architectures on the same split and seed"):
        st.caption("These experiments train new models and may take time, especially on CPU.")
        s1, s2 = st.columns(2)
        with s1:
            preset=st.selectbox("Experiment set", ["Quick comparison (2 configurations)", "Three-model comparison"], key="sweep_preset")
        with s2:
            compare_iters=st.select_slider("Iterations per comparison model", options=[100,200,500,1000], value=200, key="sweep_iters")
        if st.button("Run parameter comparison", type="primary", key="sweep_cta"):
            presets=[(64,4,2),(128,4,4)] if preset.startswith("Quick") else [(64,4,2),(128,4,4),(256,8,6)]
            rows=[]
            bar=st.progress(0)
            for i,(e,h,l) in enumerate(presets):
                with st.spinner(f"Training configuration {i+1}/{len(presets)}: emb={e}, heads={h}, layers={l}"):
                    cr=train_model(corpus,e,h,l,block,dropout,batch,compare_iters,lr,max(1,compare_iters//5),int(seed))
                    cm=test_metrics(cr)
                    rows.append({"Model":f"emb{e}-head{h}-layer{l}","Embedding":e,"Heads":h,"Layers":l,
                                 "Iterations":compare_iters,"Parameters":cr["params"],"Train time (s)":round(cr["seconds"],2),
                                 "Train loss":round(cr["history"]["train"][-1],4),"Validation loss":round(cr["history"]["val"][-1],4),
                                 "Held-out loss":round(cm["loss"],4),"Perplexity":round(cm["perplexity"],3),
                                 "Accuracy (%)":round(cm["accuracy"]*100,2)})
                bar.progress((i+1)/len(presets))
            st.session_state.comparisons=rows

    if st.session_state.comparisons:
        df=pd.DataFrame(st.session_state.comparisons)
        with ui_card("Comparison Results", "All architecture runs from this session"):
            st.dataframe(df, hide_index=True)
            st.download_button("Download comparison CSV",df.to_csv(index=False).encode(),"parameter_comparison.csv","text/csv")
        with ui_card("Architecture Comparison", "Select a metric to compare architectures"):
            metric_choice=st.selectbox("Plot metric", ["Validation loss","Perplexity","Accuracy (%)","Parameters","Train time (s)"])
            ui_sensitivity_chart(df, metric_choice, figsize=(9.2, 3.4))
    else:
        with ui_card("Architecture Comparison", "No comparison runs yet"):
            st.caption("Run a comparison to see architecture-level validation loss, perplexity, accuracy, parameters, and training time.")

with tab_conf:
    if r is None:
        ui_empty_tab("Confusion Matrix")
    else:
        unit = "token" if metrics.get("token_based") else "character"
        if len(metrics["y_true"]) == 0:
            with ui_card(f"Next-{unit.title()} Confusion Matrix", "Not available for this checkpoint"):
                st.info(
                    "This checkpoint has no validation.txt long enough for local recomputation, so "
                    "per-token predictions are unavailable. Aggregate loss, perplexity, and accuracy "
                    "shown in Training History come from metadata.json."
                )
        else:
            max_display = max(2, min(30, len(r["chars"])))
            default_display = min(15, max_display)
            with ui_card("Matrix Controls", f"Top-{unit} preview settings"):
                st.caption(f"The plot shows the most frequent actual {unit}s in this evaluated subset; all others are grouped as 'other'.")
                top_n=st.slider("Number of frequent groups to display", 5, max_display, default_display)
            with ui_card(f"Row-normalized Next-{unit.title()} Confusion Matrix", "Actual vs predicted groups"):
                true,pred=metrics["y_true"],metrics["y_pred"]
                counts=np.bincount(true,minlength=len(r["chars"]))
                top_ids=np.argsort(counts)[::-1][:top_n]
                mapping={int(v):i for i,v in enumerate(top_ids)}
                other_idx=len(top_ids)
                yt=np.array([mapping.get(int(v),other_idx) for v in true])
                yp=np.array([mapping.get(int(v),other_idx) for v in pred])
                labels=[repr(r["itos"][int(i)]) for i in top_ids]+["other"]
                cm=confusion_matrix(yt,yp,labels=list(range(len(labels))),normalize="true")
                fig,ax=plt.subplots(figsize=(9.2,4.8),dpi=110)
                fig.patch.set_facecolor("#FFFFFF")
                sns.heatmap(cm,xticklabels=labels,yticklabels=labels,cmap="Oranges",ax=ax,annot=False,cbar_kws={"shrink":0.8},linewidths=0.4,linecolor="#FFFFFF")
                ax.set_xlabel(f"Predicted {unit}", color="#57534E", fontsize=10)
                ax.set_ylabel(f"Actual {unit}", color="#57534E", fontsize=10)
                ax.tick_params(labelsize=8, colors="#57534E")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Rows are normalized, so each row shows the distribution of predictions for that actual group.")
            with ui_card(f"Precision / Recall / F1 by {unit.title()} Group", f"Per-{unit} error analysis"):
                report=classification_report(yt,yp,labels=list(range(len(labels))),target_names=labels,zero_division=0,output_dict=True)
                rep=pd.DataFrame(report).T.reset_index().rename(columns={"index":"Class"})
                st.dataframe(rep)
                st.download_button("Download classification report CSV",rep.to_csv(index=False).encode(),"classification_report.csv","text/csv")

st.html('<div class="footnote">Academic note: all displayed metrics are computed from the model run in this session. The held-out 10% split is called validation in the original notebook; for a rigorous final report, reserve a separate test split before final evaluation.</div>')
