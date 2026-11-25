# Mini-GPT: Colab GPU training → local Streamlit integration

This document explains how to train the project's **own custom Transformer** on Google Colab and load the exported checkpoint in the existing local app. No external LLM/API and no pretrained language-model weights are used.

## What changed in the repository

| File | Purpose |
| --- | --- |
| `MiniGPT_Colab_Training.ipynb` | Complete 15-section Colab notebook: environment, GPU check, streaming datasets, cleaning, splits, BPE, custom Transformer, training, evaluation, generation, export, optional SFT, verification. |
| `minigpt_colab.py` | Single source of truth for the Transformer, byte-level BPE tokenizer, checkpoint save/load/validation, and evaluation helpers. Embedded verbatim in the notebook and imported by `app.py`. |
| `app.py` | Adds a **Colab checkpoint (optional)** loader. Preserves the original character-level training/evaluation/generation path and the existing Streamlit UI. |
| `COLAB_SETUP.md` | This setup/verification guide. |
| `requirements.txt` | Unchanged. The local app needs no new packages; the notebook installs `datasets`, `pyarrow`, and `tqdm` itself. |
| `input.txt` | Unchanged. |

Shared module fingerprint after the prompt/generation fix: `2d77195528ea809d`. The notebook prints this after import; use it to confirm Colab and local code are synchronized.

The bundled `mini_gpt_trained_model/metadata.json` still records the pre-fix fingerprint `6310b3745d82156b` because it was exported before the inference fix. The architecture, vocabulary, and state-dict keys still match exactly; re-running the notebook will refresh that recorded fingerprint.

## 1. Local setup

```bash
cd /path/to/Mini-Transformer
python -m pip install -r requirements.txt
streamlit run app.py
```

The original app still works exactly as before. `input.txt` remains the default Tiny Shakespeare corpus, and the **Train & Evaluate Model**, **Reset**, and **Run parameter comparison** controls are unchanged.

## 2. Colab setup

1. Open <https://colab.research.google.com/>.
2. Upload `MiniGPT_Colab_Training.ipynb` from this repository, or open it from GitHub.
3. Select **Runtime → Change runtime type → GPU**.
4. Run all cells in order.
5. The first cell mounts Google Drive. Accept the authorization prompt.
6. The notebook prints the actual GPU name, memory, selected architecture, parameter count, dataset counts, token counts, loss curves, perplexity, and real generated text.

### Hardware-based defaults

The notebook reads actual CUDA hardware before choosing settings:

| Detected GPU memory | `n_embd` | heads | layers | block | batch | iters | vocab |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ≥ 16 GiB | 256 | 8 | 6 | 256 | 32 | 6000 | 4096 |
| ≥ 8 GiB | 192 | 6 | 4 | 192 | 16 | 4000 | 4096 |
| CPU / < 8 GiB | 128 | 4 | 4 | 128 | 16 | 2000 | 2048 |

Edit `MODEL_CFG` in the GPU-detection cell if you want a different benchmark. Increase capacity only after observing real memory headroom and step time.

## 3. Datasets

The pipeline streams configurable subsets — it never downloads all of FineWeb-Edu.

| Source | Dataset | Subset | License recorded in metadata |
| --- | --- | --- | --- |
| TinyStories | `roneneldan/TinyStories` | first N train records | `cdla-sharing-1.0` |
| WikiText-103 | `Salesforce/wikitext` | `wikitext-103-v1` | see dataset card |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | `odc-by` |

Default configuration in the notebook:

```python
DATASET_CONFIG = {
    "tinystories": {"enabled": True, "max_docs": 25000},
    "wikitext103": {"enabled": True, "max_docs": 8000},
    "fineweb_edu": {"enabled": False, "max_docs": 3000},
}
```

Cleaning keeps English spelling/punctuation and `\n\n` paragraph boundaries, removes control characters, normalizes quotes/dashes, drops empty/short paragraphs, and deduplicates records. Whole documents are assigned to train/val/test, and the notebook asserts that no document hash appears in more than one split.

## 4. Tokenizer and architecture

- Byte-level BPE trained **on the training split only**.
- Special tokens: `<pad>=0`, `<bos>=1`, `<eos>=2`, `<unk>=3`.
- Target vocabulary ≈ 4096 tokens including 256 byte fallback tokens.
- Round-trip tests run on unseen English and Unicode sentences.
- `MODEL_CFG["vocab_size"]` is set from the trained tokenizer, so model output dimensions always match the vocabulary.
- Architecture: token + positional embeddings, causal multi-head self-attention with Q/K/V projections, feed-forward network, residual connections, LayerNorm, and LM head — the same state-dict layout as the original app.

## 5. Exported artifacts

After a successful Colab run, `mini_gpt_trained_model/` contains:

```text
mini_gpt_trained_model/
  model.pt                  # best-validation state dict + config
  config.json               # architecture config
  tokenizer.json            # BPE vocab + merges
  training_metrics.csv      # step/train/val/lr/elapsed_seconds
  metadata.json             # dataset, tokenizer, architecture, training, evaluation, generation
  sample_generations.txt    # real before/after generations
  validation.txt            # optional: lets the local app recompute token-level metrics
```

The notebook copies this folder to `/content/drive/MyDrive/mini_gpt_trained_model` and then reloads it into fresh model/tokenizer objects on CPU as a self-containment check.

## 6. Local integration

1. Download `mini_gpt_trained_model/` from Google Drive.
2. Put the folder next to `app.py`:

   ```text
   Mini-Transformer/
     app.py
     minigpt_colab.py
     MiniGPT_Colab_Training.ipynb
     input.txt
     requirements.txt
     mini_gpt_trained_model/
       model.pt
       config.json
       tokenizer.json
       training_metrics.csv
       metadata.json
       sample_generations.txt
       validation.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

4. In **Model Hyperparameters → Colab checkpoint (optional)**, keep `mini_gpt_trained_model` and click **Load checkpoint**.
5. Use the **Text Generation** tab for the editable prompt, temperature, top-k, top-p, repetition penalty, no-repeat n-gram, max tokens, and EOS stopping. The prompt is a `text_area`; your custom text persists across reruns. Generation encodes it as `<bos>` + text (no trailing `<eos>`) and displays only the newly generated continuation, with the prompt and token count shown above the output.
6. Click **Reset** to return to the original character-level session model. Nothing on disk is deleted.

### Validation performed on load

The loader checks:

- all required files exist;
- `config.json` has required architecture keys;
- `n_embd % n_head == 0`;
- tokenizer vocabulary size equals `config.json` vocab size;
- state-dict keys exactly match the reconstructed architecture;
- metadata is read for recorded loss/perplexity/accuracy.

Failures appear as clear Streamlit errors (for example, a missing-file list), not raw tracebacks.

## 7. Actual verification completed here

The following tests **were run** in the local repository using `/opt/anaconda3/bin/python` and Streamlit `AppTest`:

| Test | Result |
| --- | --- |
| `py_compile app.py minigpt_colab.py` | passed |
| AST scan for unsupported `st.*(..., width=...)` arguments | 0 matches |
| App initial load | 0 exceptions; buttons: Train & Evaluate Model, Reset, Load checkpoint, Run parameter comparison; 5 tabs rendered |
| Legacy path: click Train & Evaluate Model | 0 exceptions |
| Legacy path: click Generate text | 0 exceptions; 306 characters generated from `ROMEO:` |
| BPE round-trip on unseen English + Unicode sentences | 3/3 passed |
| Checkpoint save → fresh load → generate | passed with a local smoke-test artifact (vocab 300, 45,804 parameters) |
| Missing checkpoint folder | clear error listing `model.pt`, `config.json`, `tokenizer.json`; 0 exceptions |
| Notebook training/export logic dry run | passed on tiny synthetic CPU data, 12 iterations: train loss 5.9190 → 5.7440, val loss 5.9495 → 5.7213, test loss 5.6933, perplexity 296.8643, duration 0.16 s |

The tiny dry-run numbers above are **code-path smoke-test results only**. They are not training results for your project and must not be reported as such.

## 8. Results that can only come from your Colab run

I do not have access to your Google Colab runtime or GPU from this environment, so the following are **not yet produced** and have not been fabricated:

- GPU model actually used;
- dataset document counts and token counts for your selected caps;
- final vocabulary size after training on your data;
- parameter count for the selected GPU configuration;
- real training duration;
- real train/validation/test loss and perplexity;
- real generation samples for the six requested prompts.

Run the notebook to fill these values. `metadata.json`, `training_metrics.csv`, and the notebook output contain the authoritative numbers.

## 9. Optional instruction fine-tuning

The notebook includes a separate, disabled-by-default stage (`ENABLE_SFT = False`) using `databricks/databricks-dolly-15k`. It starts from the pretrained checkpoint, uses the same tokenizer/architecture, trains briefly at a lower learning rate, and saves to `mini_gpt_sft_model/` without overwriting the base export. SFT does **not** guarantee factual accuracy.

## 10. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Checkpoint folder is incomplete...` | Put all required files in one folder and paste that exact folder path into the app. |
| `Vocabulary mismatch...` | The checkpoint and tokenizer came from different runs. Re-export both together. |
| `Checkpoint architecture does not match config.json` | Do not edit `config.json` manually; re-export from the notebook. |
| `Input context exceeds block size` | Lower max tokens or increase `block_size` in Colab and retrain. |
| Output shows only the prompt | This was caused by encoding the inference prompt with a trailing `<eos>` and decoding the full sequence. It is fixed: prompts are BOS-only and only new tokens are decoded. Reload the app. |
| App loads but output is random | The smoke-test/random checkpoint may have been used. Load a real Colab artifact from `mini_gpt_trained_model/`. |
| Out of memory in Colab | Lower `batch_size`, `block_size`, `n_layer`, or `n_embd` in `MODEL_CFG`. |
| Slow local generation | Select CPU/GPU appropriately; local CPU inference is expected to be slower than Colab. |
