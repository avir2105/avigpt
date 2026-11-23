# Colab serve notebook (cells)

Copy these cells into a Colab notebook with **GPU (T4)**. Expose the API with **ngrok** only.

## Cell 1 — setup

```python
!git clone https://github.com/avir2105/avigpt.git
%cd avigpt
!pip install -q -r requirements.txt
!nvidia-smi
```

If the repo layout is `avigpt/avigpt`, `cd` into the inner folder that contains `app.py`.

## Cell 2 — start API

```python
import os, subprocess, time, requests
os.environ["MODEL_NAME"] = "avi2135/avigpt2"
os.environ["MAX_NEW_TOKENS"] = "128"
os.environ["HOST"] = "0.0.0.0"
os.environ["PORT"] = "8000"
subprocess.Popen(["python", "-m", "serve.app"])
for _ in range(120):
    try:
        if requests.get("http://127.0.0.1:8000/health", timeout=2).ok:
            print(requests.get("http://127.0.0.1:8000/health").json())
            break
    except Exception:
        pass
    time.sleep(2)
else:
    raise RuntimeError("API not ready")
```

## Cell 3 — ngrok

```python
!pip install -q pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")  # https://dashboard.ngrok.com/get-started/your-authtoken
print(ngrok.connect(8000, "http"))
```

## Cell 4 — point local UI

Open `ui/index.html` on your laptop → paste the ngrok URL → chat / verify.
