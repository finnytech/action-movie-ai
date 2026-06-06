# Action Movie AI Studio (L40S / Lightning AI Edition)

This directory is set up on the **D: Drive (HDD)** to host the Action Movie Video & Audio AI Generator workspace.

## Drive Choice
- **Drive:** `D:` (HDD)
- **Path:** `D:/workspace/action_movie_ai/`
- Designed to save storage space on your primary `C:` SSD while maintaining high performance.

## Hardware & Environment Setup
- **GPU:** NVIDIA L40S (48 GB VRAM)
- **Platform:** Lightning AI Studio (Not Google Colab)
- **Video Model:** **HunyuanVideo (13B)** (Distilled 1.5 default)
- **Audio Model:** **AudioLDM 2 Large (1.5B)**

## Files
- `lightning_studio_script.py` — The core Python script that runs the Gradio web UI.
- `README.md` — This file.

## How to Run in a Lightning AI Studio
1. Open a terminal in your Lightning AI Studio.
2. Run the script using Python:
   ```bash
   python D:/workspace/action_movie_ai/lightning_studio_script.py
   ```
3. Once running, Gradio will expose the web application locally on port `7860`.
4. Lightning AI will automatically detect the running server on port `7860` and show a secure public link or button (e.g. "Open Web Space" or similar) in the top-right of your Studio dashboard to access the Gradio interface directly in your browser.
5. In the Gradio UI:
   - Enter your video prompt and audio prompt.
   - Click "🔥 ACTION!" to start the generation.
   - The video and sound will be generated sequentially (to prevent system RAM OOM crash) and merged using FFmpeg.
   - The final video `action_movie_scene.mp4` will be played on the preview screen and saved in the workspace.
