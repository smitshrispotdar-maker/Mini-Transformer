# Mini-Transformer Evaluation Lab

Streamlit GUI for the Tiny Shakespeare character-level MiniGPT from your GENAI.ipynb.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

If automatic dataset download fails, download Tiny Shakespeare's `input.txt` and upload it using the sidebar file uploader.

## Tabs
- Overview: configuration, parameter count, training/validation loss
- Training & Evaluation: held-out loss, perplexity, next-character accuracy
- Text Generation: prompt, temperature and top-k
- Parameter Comparison: train and compare different model sizes
- Confusion Matrix: normalized character prediction matrix and classification report

Metrics are calculated from each actual run; no sample values are hard-coded. The original notebook has a 90/10 train/validation split, so the app labels the second portion as held-out validation rather than claiming it is an independent test set.
