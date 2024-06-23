
# Streamlit Chat

## Setup

* create `.streamlit/secrets.toml` under `$HOME` or current folder.

```toml
# secrets.toml
OPENAI_API_KEY = "sk-..."
```

* Conda setup:

```bash
conda create -n <env> python=3.10
conda activate <env>

pip install -r requirements.txt
streamlit run app.py
```

* Docker

```bash
docker build . -t st-chat
docker run -p 8501:8501 -v .:/app st-chat
```

