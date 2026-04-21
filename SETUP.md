# Claude Code MLX Proxy - Setup Guide

This guide explains how to properly configure your terminal environment and launch the proxy so that the Anthropic `claude-code` CLI natively leverages your local `gemma-4` engine.

### 1. Installation & Environment Patching
Because running `gemma-4` natively on Apple Silicon requires specific MLX 0.31 kernel updates missing from upstream packages, you must apply the repository's native patch script first:

```bash
uv sync
./patch_env.sh
```

### 2. Launching the Proxy Server
Before launching Claude Code, you must ensure the background HTTP proxy is actively running and ready to handle prompts. 

In your terminal, run:
```bash
uv run main.py
```
*Tip: Wait until your terminal says `Uvicorn running on http://0.0.0.0:8888` and the model is successfully loaded into memory before proceeding!*

### 3. Masking Claude's Environment Variables
Open a **new, separate terminal tab** (leave the proxy running in the background!). 

To force Claude to route API traffic directly to your machine instead of the cloud, export these variables:
```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8888"
export ANTHROPIC_API_KEY="dummy"
```

### 4. Running the CLI
Run Claude exactly as you normally would:
```bash
claude
```

> [!WARNING]
> Claude will likely display a prompt warning you that it **"Detected a custom API key in your environment."** 
> 
> You **MUST select `1. Yes`**. If you select "No", the CLI will ignore your environment variables and default back to the cloud APIs.

### Optional: Permanent Persistence
If you exclusively use the local proxy and never intend to connect Claude to the cloud, you can permanently register the routing variables to your Mac's Z-Shell profile:
```bash
echo 'export ANTHROPIC_BASE_URL="http://127.0.0.1:8888"' >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY="dummy"' >> ~/.zshrc
source ~/.zshrc
```
