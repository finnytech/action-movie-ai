# ==============================================================================
# HUNYUANVIDEO & AUDIOLDM2 ACTION STUDIO - LIGHTNING AI (L40S EDITION)
# Optimized for NVIDIA L40S / A100 GPUs on Lightning AI Studio platform
# ==============================================================================
# Save this file as `lightning_studio_script.py` on your D: drive:
# D:/workspace/action_movie_ai/lightning_studio_script.py
#
# Run it in your Lightning AI terminal:
# python D:/workspace/action_movie_ai/lightning_studio_script.py

import subprocess
import sys

print("Checking and installing required libraries. Please wait...")
# We do not upgrade 'torch' by default because upgrading it in Lightning AI conda environment breaks torchvision compatibility.
# Instead, we install other packages and ensure torchvision/torch are compatible.
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-U",
    "diffusers", "transformers", "accelerate", "scipy", 
    "gradio", "sentencepiece", "protobuf", "imageio-ffmpeg"
], check=True)

# Repair torchvision mismatch if present
try:
    import torchvision
except Exception as e:
    err_msg = str(e)
    if "torchvision::nms" in err_msg or "torchvision" in err_msg or "operator" in err_msg:
        print("Detected torchvision/torch mismatch. Repairing and restoring torch==2.8.0...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.8.0"], check=True)

print("Libraries check complete!")

import os
import gc
import torch
import scipy.io.wavfile as wav
from diffusers import HunyuanVideoPipeline, AudioLDM2Pipeline
from diffusers.utils import export_to_video
import gradio as gr

def clear_memory():
    """Aggressively clears system RAM and GPU VRAM to prevent Out-Of-Memory crashes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[Memory Cleared] GPU VRAM and System RAM emptied.")

def generate_action_scene(
    video_prompt,
    video_negative_prompt,
    audio_prompt,
    model_name,
    resolution,
    num_frames,
    fps,
    video_steps,
    cfg_scale,
    seed,
    audio_steps,
    enable_cpu_offload
):
    clear_memory()
    
    # Resolve resolution (width x height)
    try:
        width_str, height_str = resolution.split("x")
        width = int(width_str)
        height = int(height_str)
    except ValueError:
        width, height = 720, 480  # fallback

    seed = int(seed)
    num_frames = int(num_frames)
    fps = int(fps)
    video_steps = int(video_steps)
    audio_steps = int(audio_steps)

    temp_video_path = "temp_silent.mp4"
    temp_audio_path = "temp_audio.wav"
    output_filename = "action_movie_scene.mp4"

    # Clean up previous runs
    for path in [temp_video_path, temp_audio_path, output_filename]:
        if os.path.exists(path):
            os.remove(path)

    # ==========================================
    # PHASE 1: GENERATE VIDEO (HUNYUANVIDEO)
    # ==========================================
    print(f"\n--- Loading Video Model: {model_name} ---")
    try:
        # Load the pipeline in bfloat16 (native for HunyuanVideo)
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        
        # Apply VRAM optimizations
        pipe.vae.enable_tiling()
        
        if enable_cpu_offload:
            pipe.enable_model_cpu_offload()
            print("CPU Offload enabled for Video Model.")
        else:
            pipe.to("cuda")
            print("Loaded Video Model directly into GPU VRAM.")

        # Setup Seed
        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None

        print("Generating video frames (this may take 1-3 minutes)...")
        video_frames = pipe(
            prompt=video_prompt,
            negative_prompt=video_negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=video_steps,
            guidance_scale=cfg_scale,
            generator=generator
        ).frames[0]

        print("Exporting silent video...")
        export_to_video(video_frames, temp_video_path, fps=fps)
        print("Video generation phase complete!")

    except Exception as e:
        print(f"Error during video generation: {e}")
        return None, f"Video Error: {str(e)}"
    
    finally:
        # Crucial clean-up to free RAM/VRAM before loading the Audio model
        if 'pipe' in locals():
            del pipe
        clear_memory()

    # ==========================================
    # PHASE 2: GENERATE AUDIO (AUDIOLDM2)
    # ==========================================
    print("\n--- Loading Audio Model: cvssp/audioldm2-large ---")
    try:
        audio_pipe = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2-large",
            torch_dtype=torch.float16
        )
        
        if enable_cpu_offload:
            audio_pipe.enable_model_cpu_offload()
            print("CPU Offload enabled for Audio Model.")
        else:
            audio_pipe.to("cuda")
            print("Loaded Audio Model directly into GPU VRAM.")

        # Calculate exact audio duration in seconds
        audio_duration = num_frames / fps
        print(f"Generating audio for {audio_duration:.2f} seconds based on audio prompt...")

        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        audio_output = audio_pipe(
            prompt=audio_prompt,
            negative_prompt="low quality, ambient noise, static, hiss, music, bad quality",
            num_inference_steps=audio_steps,
            audio_length_in_s=audio_duration,
            generator=generator
        ).audios[0]

        print("Saving audio track...")
        # Write wav with 16000Hz sampling rate
        wav.write(temp_audio_path, rate=16000, data=audio_output)
        print("Audio generation phase complete!")

    except Exception as e:
        print(f"Error during audio generation: {e}")
        # If audio fails, we can still return the silent video
        if os.path.exists(temp_video_path):
            os.rename(temp_video_path, output_filename)
            return output_filename, f"Warning: Audio failed ({str(e)}). Showing silent video."
        return None, f"Audio Error: {str(e)}"
        
    finally:
        if 'audio_pipe' in locals():
            del audio_pipe
        clear_memory()

    # ==========================================
    # PHASE 3: MERGE VIDEO AND AUDIO VIA FFMPEG
    # ==========================================
    print("\n--- Merging video and audio using FFmpeg ---")
    if os.path.exists(temp_video_path) and os.path.exists(temp_audio_path):
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video_path,
                "-i", temp_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_filename
            ]
            # Execute command with standard output suppressed to avoid clutter
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print("Merge successful!")
        except Exception as e:
            print(f"FFmpeg merging failed: {e}")
            # Fallback to silent video if merging fails
            os.rename(temp_video_path, output_filename)
            return output_filename, f"Warning: FFmpeg merge failed. Showing silent video."
        finally:
            # Clean up temp files
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    else:
        return None, "Error: Generation succeeded but temporary files were missing."

    print("Successfully generated final action video with high-fidelity audio!")
    return output_filename, "Success! Your cinematic action video is ready."

# ==========================================
# PHASE 4: GRADIO WEB UI INTERFACE
# ==========================================
css = """
body { background-color: #0b0f19; color: #f3f4f6; font-family: 'Inter', sans-serif; }
.gradio-container { border-radius: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4); }
#title-header { text-align: center; margin-bottom: 20px; }
#title-header h1 { font-weight: 800; background: linear-gradient(90deg, #ff4e50, #f9d423); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="amber"), css=css) as demo:
    with gr.Row(elem_id="title-header"):
        gr.Markdown(
            """
            # 🎬 HUNYUANVIDEO & AUDIOLDM2 ACTION STUDIO
            ### Cinematic Action Video (PG-16) & Sound Design Generator
            *Optimized for NVIDIA L40S & A100 GPUs on Lightning AI. Runs sequentially to protect system RAM.*
            """
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Regie-Anweisungen")
            video_prompt = gr.Textbox(
                label="Video Prompt (Englisch empfohlen)",
                placeholder="A high-octane motorcycle chase through neon-lit streets of Tokyo. Sparks flying, realistic lighting, camera tracking behind.",
                value="A cinematic action fight scene of two warriors in a rainy alleyway at night. Slow motion punches, splashing water, dramatic cinematic lighting, photorealistic.",
                lines=3
            )
            video_neg_prompt = gr.Textbox(
                label="Negative Prompt (Was du NICHT willst)",
                value="blurry, worst quality, low quality, static, deformed, cartoon, 3d, anime",
                lines=2
            )
            audio_prompt = gr.Textbox(
                label="Audio Design Prompt (Wie soll die Szene klingen?)",
                placeholder="loud thunder, heavy rain, metal impacts, grunts, intense background cinematic drums",
                value="heavy rain, loud thunder rumbling, physical punches, grunts, cinematic action score",
                lines=2
            )
            
            with gr.Accordion("⚙️ Kamera- & Modell-Optionen", open=True):
                model_name = gr.Dropdown(
                    label="HunyuanVideo Modell",
                    choices=[
                        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
                        "hunyuanvideo-community/HunyuanVideo"
                    ],
                    value="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled"
                )
                resolution = gr.Dropdown(
                    label="Auflösung",
                    choices=["720x480", "854x480", "960x544", "1280x720"],
                    value="720x480"
                )
                num_frames = gr.Slider(
                    label="Anzahl Frames (Länge)",
                    minimum=17,
                    maximum=129,
                    step=16,
                    value=49,
                    info="Formel: 4k + 1. 49 Frames = ~3 Sek bei 15fps, 81 Frames = ~5 Sek bei 15fps"
                )
                fps = gr.Slider(
                    label="FPS (Geschwindigkeit)",
                    minimum=8,
                    maximum=30,
                    step=1,
                    value=15
                )
                video_steps = gr.Slider(
                    label="Video Qualitäts-Steps",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=15,
                    info="Distilled-Modell: 10-15 Steps. Base-Modell: 30-50 Steps."
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale (Prompt-Treue)",
                    minimum=1.0,
                    maximum=15.0,
                    step=0.5,
                    value=6.0
                )
                seed = gr.Number(
                    label="Seed (-1 für Zufall)",
                    value=-1,
                    precision=0
                )
                audio_steps = gr.Slider(
                    label="Audio-Qualitäts-Steps",
                    minimum=20,
                    maximum=200,
                    step=10,
                    value=100
                )
                enable_cpu_offload = gr.Checkbox(
                    label="Agressives CPU-Offloading aktivieren",
                    value=False,
                    info="Aktivieren, falls der VRAM überläuft (z.B. bei 1280x720 Auflösung)."
                )

            generate_btn = gr.Button("🔥 ACTION! (Generieren)", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📺 Preview-Monitor")
            status_output = gr.Textbox(label="Status-Meldungen", interactive=False)
            video_output = gr.Video(
                label="Dein fertiger Action-Film",
                interactive=False,
                autoplay=True,
                loop=True
            )
            
    generate_btn.click(
        fn=generate_action_scene,
        inputs=[
            video_prompt,
            video_neg_prompt,
            audio_prompt,
            model_name,
            resolution,
            num_frames,
            fps,
            video_steps,
            cfg_scale,
            seed,
            audio_steps,
            enable_cpu_offload
        ],
        outputs=[video_output, status_output]
    )

if __name__ == "__main__":
    # Standard launch configuration for Lightning AI Studio
    # exposing web server on port 7860 so Lightning AI port forwarder detects it automatically
    demo.launch(server_name="0.0.0.0", server_port=7860)
