# ==============================================================================
# HUNYUANVIDEO & AUDIOLDM2 ACTION STUDIO - LIGHTNING AI (L40S EDITION)
# Optimized for NVIDIA L40S / A100 GPUs on Lightning AI Studio platform
# ==============================================================================
# Save this file as `lightning_studio_script.py` on your D: drive:
# D:/workspace/action_movie_ai/lightning_studio_script.py
#
# Run it in your Lightning AI terminal:
# python D:/workspace/action_movie_ai/lightning_studio_script.py

import os
import gc
import sys
import subprocess

print("Checking and installing required libraries. Please wait...")
# We install required packages without upgrading dependencies to avoid breaking Lightning AI's pre-installed torch/torchvision
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "diffusers", "transformers", "accelerate", "scipy", 
    "gradio", "sentencepiece", "protobuf", "imageio-ffmpeg", "huggingface_hub"
], check=True)

print("Libraries check complete!")

import torch
import torchvision
import scipy.io.wavfile as wav
from diffusers import HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline, AudioLDM2Pipeline
from diffusers.utils import export_to_video
import gradio as gr
from huggingface_hub import snapshot_download

# ==========================================
# PHASE 0: PRE-DOWNLOAD MODELS
# ==========================================
def pre_download_models():
    print("\n--- Checking and Pre-Downloading Models ---")
    print("This ensures all huge files are on the hard drive before the app starts.")
    models = [
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled",
        "cvssp/audioldm2-large"
    ]
    for model_id in models:
        print(f"Verifying {model_id}...")
        snapshot_download(repo_id=model_id)
        print(f"✅ {model_id} is fully downloaded and ready!")
    print("All models are cached locally!\n")

pre_download_models()

def clear_memory():
    """Aggressively clears system RAM and GPU VRAM to prevent Out-Of-Memory crashes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[Memory Cleared] GPU VRAM and System RAM emptied.")

def concat_videos(video_paths, output_path):
    if len(video_paths) == 1:
        import shutil
        shutil.copy2(video_paths[0], output_path)
        return
    # create text file for ffmpeg concat
    list_file = "concat_list.txt"
    with open(list_file, "w") as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    os.remove(list_file)

def generate_action_scene(
    video_prompt,
    audio_prompt,
    model_name,
    resolution,
    target_duration,
    fps,
    video_steps,
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
    target_duration = int(target_duration)
    fps = int(fps)
    video_steps = int(video_steps)
    audio_steps = int(audio_steps)

    # 49 frames at 15 fps = ~3.26s per chunk
    frames_per_chunk = 49  
    chunk_duration = frames_per_chunk / fps
    num_chunks = max(1, int(target_duration / chunk_duration) + (1 if target_duration % chunk_duration > 0 else 0))

    temp_audio_path = "temp_audio.wav"
    final_output_filename = "action_movie_scene.mp4"
    current_merged_video = "current_merged.mp4"

    # Clean up previous runs
    for path in [temp_audio_path, final_output_filename, current_merged_video]:
        if os.path.exists(path):
            os.remove(path)

    generated_chunk_paths = []
    last_frame_image = None

    # ==========================================
    # PHASE 1: GENERATE VIDEO CHUNKS
    # ==========================================
    for i in range(num_chunks):
        chunk_path = f"chunk_{i}.mp4"
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
            
        print(f"\n--- Generating Video Chunk {i+1}/{num_chunks} ---")
        
        try:
            if i == 0:
                print(f"Loading Text-to-Video Model: {model_name}")
                pipe = HunyuanVideo15Pipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
                pipe.vae.enable_tiling()
                if enable_cpu_offload:
                    pipe.enable_model_cpu_offload()
                    print("CPU Offload enabled for T2V Model.")
                else:
                    pipe.to("cuda")
                    print("Loaded T2V Model directly into GPU VRAM.")

                generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
                
                print(f"Generating first chunk...")
                video_frames = pipe(
                    prompt=video_prompt,
                    height=height,
                    width=width,
                    num_frames=frames_per_chunk,
                    num_inference_steps=video_steps,
                    generator=generator
                ).frames[0]

            else:
                i2v_model = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled"
                print(f"Loading Image-to-Video Model for Continuation: {i2v_model}")
                pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(i2v_model, torch_dtype=torch.bfloat16)
                pipe.vae.enable_tiling()
                if enable_cpu_offload:
                    pipe.enable_model_cpu_offload()
                    print("CPU Offload enabled for I2V Model.")
                else:
                    pipe.to("cuda")
                    print("Loaded I2V Model directly into GPU VRAM.")

                generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None

                print(f"Generating continuation chunk from previous frame...")
                video_frames = pipe(
                    image=last_frame_image,
                    prompt=video_prompt,
                    height=height,
                    width=width,
                    num_frames=frames_per_chunk,
                    num_inference_steps=video_steps,
                    generator=generator
                ).frames[0]

            print(f"Exporting chunk {i+1}...")
            export_to_video(video_frames, chunk_path, fps=fps)
            generated_chunk_paths.append(chunk_path)
            last_frame_image = video_frames[-1]  # Save the last PIL Image for the next chunk

        except Exception as e:
            err_msg = f"Error during video generation at chunk {i+1}: {e}"
            print(err_msg)
            yield current_merged_video if os.path.exists(current_merged_video) else None, err_msg, None
            return
        
        finally:
            if 'pipe' in locals():
                del pipe
            clear_memory()

        # Merge what we have so far
        concat_videos(generated_chunk_paths, current_merged_video)
        current_len = min(target_duration, round((i+1) * chunk_duration, 1))
        yield current_merged_video, f"⏳ Generating video... ({current_len}s / {target_duration}s ready)", current_merged_video

    # ==========================================
    # PHASE 2: GENERATE AUDIO FOR TOTAL DURATION
    # ==========================================
    total_video_duration = len(generated_chunk_paths) * chunk_duration
    print(f"\n--- Loading Audio Model for {total_video_duration:.2f}s of audio ---")
    try:
        audio_pipe = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2-large",
            torch_dtype=torch.float16
        )
        if enable_cpu_offload:
            audio_pipe.enable_model_cpu_offload()
        else:
            audio_pipe.to("cuda")

        generator = torch.Generator("cuda").manual_seed(seed) if seed != -1 else None
        
        print(f"Generating {total_video_duration:.2f} seconds of audio...")
        audio_output = audio_pipe(
            prompt=audio_prompt,
            negative_prompt="low quality, ambient noise, static, hiss, music, bad quality",
            num_inference_steps=audio_steps,
            audio_length_in_s=total_video_duration,
            generator=generator
        ).audios[0]

        print("Saving audio track...")
        wav.write(temp_audio_path, rate=16000, data=audio_output)
        
    except Exception as e:
        print(f"Error during audio generation: {e}")
        yield current_merged_video, f"Warning: Audio failed ({str(e)}). Showing silent video.", current_merged_video
        return
        
    finally:
        if 'audio_pipe' in locals():
            del audio_pipe
        clear_memory()

    # ==========================================
    # PHASE 3: MERGE VIDEO AND AUDIO VIA FFMPEG
    # ==========================================
    print("\n--- Final Merge: Video + Audio ---")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", current_merged_video,
            "-i", temp_audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            final_output_filename
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        print(f"FFmpeg final merge failed: {e}")
        yield current_merged_video, "Warning: Audio merge failed. Showing silent video.", current_merged_video
        return
    finally:
        for p in generated_chunk_paths:
            if os.path.exists(p): os.remove(p)
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
        if os.path.exists(current_merged_video): os.remove(current_merged_video)

    yield final_output_filename, f"✅ Success! {total_video_duration:.1f}s cinematic action video is ready.", final_output_filename


# ==========================================
# PHASE 4: GRADIO WEB UI INTERFACE
# ==========================================
css = """
body { background-color: #0b0f19; color: #f3f4f6; font-family: 'Inter', sans-serif; }
.gradio-container { border-radius: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4); }
#title-header { text-align: center; margin-bottom: 20px; }
#title-header h1 { font-weight: 800; background: linear-gradient(90deg, #ff4e50, #f9d423); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
"""

with gr.Blocks() as demo:
    with gr.Row(elem_id="title-header"):
        gr.Markdown(
            """
            # 🎬 HUNYUANVIDEO CONTINUOUS ACTION STUDIO
            ### 30s Cinematic Action Video & Sound Design Generator
            *Auto-Chaining Image-to-Video Engine. Watch the video grow live!*
            """
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Regie-Anweisungen")
            video_prompt = gr.Textbox(
                label="Video Prompt (Englisch empfohlen)",
                value="A cinematic action fight scene of two warriors in a rainy alleyway at night. Slow motion punches, splashing water, photorealistic.",
                lines=3
            )
            audio_prompt = gr.Textbox(
                label="Audio Design Prompt",
                value="heavy rain, loud thunder rumbling, physical punches, grunts, cinematic action score",
                lines=2
            )
            
            with gr.Accordion("⚙️ Kamera- & Modell-Optionen", open=True):
                model_name = gr.Dropdown(
                    label="T2V Start-Modell",
                    choices=["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled"],
                    value="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled"
                )
                resolution = gr.Dropdown(
                    label="Auflösung",
                    choices=["720x480", "848x480"],
                    value="720x480"
                )
                target_duration = gr.Slider(
                    label="Ziel-Länge (Sekunden)",
                    minimum=4,
                    maximum=30,
                    step=4,
                    value=12,
                    info="Das Skript generiert ~3-4s Chunks und verknüpft sie nahtlos."
                )
                fps = gr.Slider(
                    label="FPS",
                    minimum=8,
                    maximum=24,
                    step=1,
                    value=15
                )
                video_steps = gr.Slider(
                    label="Video Steps pro Chunk",
                    minimum=10,
                    maximum=30,
                    step=1,
                    value=15
                )
                seed = gr.Number(label="Seed (-1 für Zufall)", value=-1, precision=0)
                audio_steps = gr.Slider(label="Audio Steps", minimum=20, maximum=100, step=10, value=50)
                enable_cpu_offload = gr.Checkbox(label="Agressives CPU-Offloading (Gegen VRAM Overflows)", value=False)

            generate_btn = gr.Button("🔥 ACTION! (Starten)", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📺 Live Preview-Monitor")
            status_output = gr.Textbox(label="Status", interactive=False)
            video_output = gr.Video(
                label="Video Stream",
                interactive=False,
                autoplay=True,
                loop=True
            )
            download_output = gr.File(label="📥 Fertiges Video Herunterladen")
            
    generate_btn.click(
        fn=generate_action_scene,
        inputs=[
            video_prompt, audio_prompt, model_name, resolution,
            target_duration, fps, video_steps, seed, audio_steps, enable_cpu_offload
        ],
        outputs=[video_output, status_output, download_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True,
        theme=gr.themes.Default(primary_hue="red", secondary_hue="amber"),
        css=css
    )
