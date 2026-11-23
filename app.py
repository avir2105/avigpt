"""Entry point — delegates to serve.app (kept for Docker / Colab compatibility)."""

from serve.app import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
