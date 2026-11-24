import math, random, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Mini-Transformer Evaluation Lab", page_icon="🧠", layout="wide")
st.title("🧠 Mini-Transformer (nanoGPT) — Evaluation Lab")
st.caption("Tiny Shakespeare · Character-level language modeling · PyTorch")

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

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        _, T, _ = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return self.dropout(F.softmax(weights, dim=-1)) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(),
                                 nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd//n_head, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=.1):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("Embedding dimension must be divisible by number of attention heads.")
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size: raise ValueError("Input context exceeds block size.")
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.final_ln(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B*T, -1), targets.reshape(B*T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=.8, top_k=0):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / max(float(temperature), 1e-5)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

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
    # Deterministic contiguous next-character evaluation on held-out validation tail.
    data = result["val_data"][:max_tokens+1]
    block = result["config"]["block_size"]
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

with st.sidebar:
    st.header("Dataset & configuration")
    upload = st.file_uploader("Optional: upload Tiny Shakespeare input.txt", type=["txt"])
    st.caption("If no file is uploaded, the app attempts to download the Tiny Shakespeare corpus.")
    emb = st.select_slider("Embedding dimension", options=[32,64,96,128,192,256], value=128)
    heads = st.select_slider("Attention heads", options=[2,4,6,8], value=4)
    layers = st.slider("Transformer layers", 1, 6, 4)
    block = st.select_slider("Context length", options=[32,64,96,128,192,256], value=128)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
    batch = st.select_slider("Batch size", options=[8,16,32,64], value=32)
    iterations = st.select_slider("Training iterations", options=[200,500,1000,2000], value=500)
    lr = st.select_slider("Learning rate", options=[0.0001,0.0003,0.0005,0.001], value=0.0003, format_func=lambda x:f"{x:g}")
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=1337)
    train_button = st.button("🚀 Train / Retrain model", type="primary", use_container_width=True)

try:
    corpus = load_corpus(upload.getvalue() if upload else None)
except Exception as e:
    st.error(str(e)); st.stop()

chars = sorted(set(corpus))
c1,c2,c3,c4 = st.columns(4)
c1.metric("Corpus characters", f"{len(corpus):,}")
c2.metric("Vocabulary size", len(chars))
c3.metric("Device", "CUDA GPU" if torch.cuda.is_available() else "CPU")
c4.metric("Train / validation split", "90% / 10%")
if train_button:
    if emb % heads != 0:
        st.error("Embedding dimension must be divisible by the number of heads.")
    elif len(corpus) < block*4:
        st.error("Corpus is too small for the selected context length.")
    else:
        with st.spinner("Training model and calculating real evaluation metrics..."):
            progress = st.progress(0, text="Training...")
            try:
                st.session_state.result = train_model(corpus, emb, heads, layers, block, dropout, batch, iterations, lr, max(1, iterations//10), int(seed))
                st.session_state.comparisons = []
                progress.progress(100, text="Training complete")
            except Exception as e:
                st.exception(e)
if st.session_state.result is None:
    st.info("Choose settings in the sidebar and click **Train / Retrain model** to populate the dashboard. The dashboard does not invent metrics.")
    st.markdown("""
    **What this app provides**
    - Training and validation loss curves
    - Test-style next-character loss, perplexity and accuracy on the held-out 10% split
    - Confusion matrix and per-character precision / recall / F1
    - Text generation with temperature and top-k sampling
    - Model configuration and parameter count
    - Repeatable parameter-comparison experiments
    """)
    st.stop()

r = st.session_state.result
metrics = test_metrics(r)
tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Overview", "📉 Training & Evaluation", "✍️ Text Generation", "⚖️ Parameter Comparison", "🔎 Confusion Matrix"])

with tab1:
    a,b,c,d = st.columns(4)
    a.metric("Trainable parameters", f"{r['params']:,}")
    b.metric("Training time", f"{r['seconds']:.1f} s")
    c.metric("Final train loss", f"{r['history']['train'][-1]:.4f}")
    d.metric("Final validation loss", f"{r['history']['val'][-1]:.4f}")
    st.subheader("Training and validation loss")
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(r["history"]["step"],r["history"]["train"],marker="o",label="Train")
    ax.plot(r["history"]["step"],r["history"]["val"],marker="o",label="Validation")
    ax.set_xlabel("Training iteration"); ax.set_ylabel("Cross-entropy loss"); ax.grid(alpha=.25); ax.legend()
    st.pyplot(fig); plt.close(fig)
    st.subheader("Current model configuration")
    st.dataframe(pd.DataFrame([{"Parameter":k,"Value":v} for k,v in r["config"].items()]), use_container_width=True, hide_index=True)
    st.caption("Validation loss is estimated from random batches. The metrics tab uses a deterministic subset of the held-out validation split.")

with tab2:
    a,b,c,d=st.columns(4)
    a.metric("Held-out cross-entropy loss", f"{metrics['loss']:.4f}")
    b.metric("Perplexity", f"{metrics['perplexity']:.3f}")
    c.metric("Next-character accuracy", f"{metrics['accuracy']*100:.2f}%")
    d.metric("Evaluated characters", f"{len(metrics['y_true']):,}")
    st.write("Perplexity = exp(cross-entropy loss). Accuracy is exact next-character argmax accuracy, not a measure of overall literary quality.")
    st.subheader("Training history")
    hist_df=pd.DataFrame(r["history"])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
    st.download_button("Download training history CSV", hist_df.to_csv(index=False).encode(), "training_history.csv", "text/csv")

with tab3:
    prompt=st.text_input("Prompt", value="ROMEO:")
    max_new=st.slider("New characters", 50, 1000, 300, 50)
    temp=st.slider("Temperature", 0.1, 1.5, 0.8, 0.1)
    top_k=st.slider("Top-k (0 = disabled)", 0, min(30,len(r["chars"])), 0)
    if st.button("Generate text"):
        if not prompt: prompt=" "
        unknown=[ch for ch in prompt if ch not in r["stoi"]]
        if unknown: st.warning(f"Prompt includes characters outside the model vocabulary: {unknown!r}. They will be skipped.")
        ids=[r["stoi"][ch] for ch in prompt if ch in r["stoi"]]
        if not ids: ids=[0]
        inp=torch.tensor([ids],dtype=torch.long,device=r["device"])
        generated=r["model"].generate(inp,max_new_tokens=max_new,temperature=temp,top_k=top_k)[0].tolist()
        st.text_area("Generated output", r["".join([])] if False else "".join(r["itos"][i] for i in generated), height=300)
        st.download_button("Download generated text", "".join(r["itos"][i] for i in generated), "generated_text.txt", "text/plain")

with tab4:
    st.write("Run small/medium/large configurations on the same corpus split and seed. These experiments train new models and may take time, especially on CPU.")
    preset=st.selectbox("Experiment set", ["Quick comparison (2 configurations)", "Three-model comparison"])
    compare_iters=st.select_slider("Iterations per comparison model", options=[100,200,500,1000], value=200)
    if st.button("Run parameter comparison"):
        presets=[(64,4,2),(128,4,4)] if preset.startswith("Quick") else [(64,4,2),(128,4,4),(256,8,6)]
        rows=[]
        bar=st.progress(0)
        for i,(e,h,l) in enumerate(presets):
            with st.spinner(f"Training configuration {i+1}/{len(presets)}: emb={e}, heads={h}, layers={l}"):
                cr=train_model(corpus,e,h,l,block,dropout,batch,compare_iters,lr,max(1,compare_iters//5),int(seed))
                cm=test_metrics(cr)
                rows.append({"Model":f"emb{e}-head{h}-layer{l}","Embedding":e,"Heads":h,"Layers":l,
                             "Parameters":cr["params"],"Train time (s)":round(cr["seconds"],2),
                             "Train loss":round(cr["history"]["train"][-1],4),"Validation loss":round(cr["history"]["val"][-1],4),
                             "Held-out loss":round(cm["loss"],4),"Perplexity":round(cm["perplexity"],3),
                             "Accuracy (%)":round(cm["accuracy"]*100,2)})
            bar.progress((i+1)/len(presets))
        st.session_state.comparisons=rows
    if st.session_state.comparisons:
        df=pd.DataFrame(st.session_state.comparisons)
        st.dataframe(df,use_container_width=True,hide_index=True)
        metric_choice=st.selectbox("Plot metric", ["Validation loss","Perplexity","Accuracy (%)","Parameters","Train time (s)"])
        fig,ax=plt.subplots(figsize=(9,4))
        ax.bar(df["Model"],df[metric_choice]); ax.set_ylabel(metric_choice); ax.tick_params(axis="x",rotation=15); ax.grid(axis="y",alpha=.25)
        st.pyplot(fig); plt.close(fig)
        st.download_button("Download comparison CSV",df.to_csv(index=False).encode(),"parameter_comparison.csv","text/csv")

with tab5:
    st.warning("A confusion matrix for character prediction can be large. The plot below shows the most frequent actual characters in this evaluated subset; all other characters are grouped as 'other' for readability.")
    true,pred=metrics["y_true"],metrics["y_pred"]
    top_n=st.slider("Number of frequent characters to display",5,min(30,len(r["chars"])),min(15,len(r["chars"])))
    counts=np.bincount(true,minlength=len(r["chars"]))
    top_ids=np.argsort(counts)[::-1][:top_n]
    # Map all non-top classes to one 'other' bucket, retaining a readable matrix.
    mapping={int(v):i for i,v in enumerate(top_ids)}
    other_idx=len(top_ids)
    yt=np.array([mapping.get(int(v),other_idx) for v in true])
    yp=np.array([mapping.get(int(v),other_idx) for v in pred])
    labels=[repr(r["itos"][int(i)]) for i in top_ids]+["other"]
    cm=confusion_matrix(yt,yp,labels=list(range(len(labels))),normalize="true")
    fig,ax=plt.subplots(figsize=(10,7))
    sns.heatmap(cm,xticklabels=labels,yticklabels=labels,cmap="magma",ax=ax)
    ax.set_xlabel("Predicted character"); ax.set_ylabel("Actual character"); ax.set_title("Row-normalized next-character confusion matrix")
    st.pyplot(fig); plt.close(fig)
    st.caption("Rows are normalized, so each row shows the distribution of predictions for that actual-character group.")
    report=classification_report(yt,yp,labels=list(range(len(labels))),target_names=labels,zero_division=0,output_dict=True)
    rep=pd.DataFrame(report).T.reset_index().rename(columns={"index":"Class"})
    st.subheader("Precision / recall / F1 by character group")
    st.dataframe(rep,use_container_width=True)
    st.download_button("Download classification report CSV",rep.to_csv(index=False).encode(),"classification_report.csv","text/csv")

st.divider()
st.caption("Academic note: all displayed metrics are computed from the model run in this session. The held-out 10% split is called validation in the original notebook; for a rigorous final report, reserve a separate test split before final evaluation.")
