#!/bin/bash
echo "Applying custom mlx_vlm patches required for MLX 0.31 and Gemma 4..."
cp -R patches/mlx_vlm/* .venv/lib/python3.13/site-packages/mlx_vlm/
echo "Patch complete! You can now run the proxy."
