"""Model-agnostic stages shared by всі diffusion-пайплайни.

prompts   — build_prompt (describe / instruct), azimuth_to_cardinal
composite — Stage 4: relight-harmonize + blend frozen vehicle (bbox-safe)
polish    — Stage 5: albumentations camera-degradation (fog/grain/noise/jpeg)
precision — VRAM → bf16 / fp8 / gguf+offload selector (Kaggle vs RunPod)
"""
