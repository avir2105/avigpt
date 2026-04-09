# Run AVIGPT in Google Colab with GPU

This guide starts the FastAPI app from [app.py](app.py) inside Google Colab using a GPU runtime.

## 1. Enable GPU in Colab

In Colab, go to `Runtime` -> `Change runtime type` -> set `Hardware accelerator` to `GPU`.

You can verify the GPU with:

```python
!nvidia-smi
```

## 2. Clone or upload the project

If the code is in GitHub, clone it:

```python
!git clone https://github.com/avir2105/avigpt.git
%cd avigpt
```

If you already uploaded the files into Colab, just change into the project folder.

## 3. Install dependencies

Run:

```python
!pip install -U pip
!pip install fastapi uvicorn[standard] transformers accelerate bitsandbytes pydantic
```

If you hit a bitsandbytes issue, restart the runtime and run the install cell again.

## 4. Start the server

The app loads the model from Hugging Face and starts FastAPI on port `8000`.

```python
import os
import subprocess

os.environ["MODEL_NAME"] = "avi2135/avigpt2"
os.environ["PORT"] = "8000"
os.environ["HOST"] = "0.0.0.0"

subprocess.Popen(["python", "app.py"])
```

You should see the model begin loading. The first startup can take a while because the model is downloaded into the Colab environment.

## 5. Expose the local API

Colab runs the app locally, so use an HTTP tunnel if you want to call it from outside Colab.

### Option A: ngrok

```python
!pip install pyngrok
```

Then run:

```python
from pyngrok import ngrok

public_url = ngrok.connect(8000)
print(public_url)
```

Use that public URL with the `/ask` endpoint.

### Option B: test directly inside Colab

You can also call the app locally from another Colab cell:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"question": "Hello, what can you do?"},
)
print(response.json())
```

## 6. Example request

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"question": "Explain quantization in one paragraph."},
)
print(response.json()["answer"])
```

## Notes

- This setup expects a GPU runtime because the model is loaded with 4-bit quantization.
- If you want to change the model, set `MODEL_NAME` before starting `app.py`.
- If Colab gives memory errors, reduce `MAX_NEW_TOKENS` or switch to a smaller model.