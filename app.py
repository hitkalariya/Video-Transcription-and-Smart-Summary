
"""
Video Transcription & Smart Summary
Author: Hit Kalariya
Email: hitkalariya88@gmail.com
GitHub: https://github.com/hitkalariya
LinkedIn: https://www.linkedin.com/in/hitkalariya/
"""

# --- Imports ---
import gradio as gr
import torch
import yt_dlp
import os
import subprocess
import json
import uuid
from transformers import AutoTokenizer, AutoModelForCausalLM
import spaces
import moviepy.editor as mp
import langdetect

# --- Model Setup ---
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
print(f"[INFO] Loading model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
print("[INFO] Model loaded successfully.")

# --- Utility Functions ---
def unique_filename(ext):
    """Generate a unique filename with the given extension."""
    return f"{uuid.uuid4()}{ext}"

def remove_files(*files):
    for f in files:
        if f and os.path.exists(f):
            os.remove(f)
            print(f"[CLEANUP] Removed: {f}")

def fetch_youtube_audio(url):
    """Download audio from a YouTube URL as .wav."""
    print(f"[DOWNLOAD] Fetching audio from: {url}")
    out_path = unique_filename(".wav")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': out_path,
        'keepvideo': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    # Sometimes yt_dlp adds .wav twice
    if os.path.exists(out_path + ".wav"):
        os.rename(out_path + ".wav", out_path)
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Audio file not found: {out_path}")
    print(f"[DOWNLOAD] Saved audio: {out_path}")
    return out_path

# --- Core Processing Functions ---
@spaces.GPU(duration=90)
def transcribe(file_path):
    """Transcribe audio or video file to text."""
    print(f"[TRANSCRIBE] File: {file_path}")
    temp_audio = None
    if file_path.lower().endswith((".mp4", ".avi", ".mov", ".flv")):
        print("[AUDIO] Extracting audio from video...")
        video = mp.VideoFileClip(file_path)
        temp_audio = unique_filename(".wav")
        video.audio.write_audiofile(temp_audio)
        file_path = temp_audio
    out_json = unique_filename(".json")
    cmd = [
        "insanely-fast-whisper",
        "--file-name", file_path,
        "--device-id", "0",
        "--model-name", "openai/whisper-large-v3",
        "--task", "transcribe",
        "--timestamp", "chunk",
        "--transcript-path", out_json
    ]
    print(f"[CMD] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Whisper failed: {e.stderr}")
        raise
    with open(out_json, "r") as f:
        data = json.load(f)
    text = data.get("text") or " ".join(chunk["text"] for chunk in data.get("chunks", []))
    remove_files(out_json)
    if temp_audio:
        remove_files(temp_audio)
    print(f"[TRANSCRIBE] Done.")
    return text

@spaces.GPU(duration=90)
def summarize(transcript):
    """Generate a summary for the given transcript."""
    print(f"[SUMMARY] Generating summary...")
    lang = langdetect.detect(transcript)
    prompt = (
        f"Summarize the following video transcription in 150-300 words.\n"
        f"The summary should be in the same language as the transcription, which is detected as {lang}.\n"
        f"Focus on the main points and key ideas.\n\n{transcript[:300000]}..."
    )
    response, _ = model.chat(tokenizer, prompt, history=[])
    print(f"[SUMMARY] Done.")
    return response

def handle_youtube(url):
    if not url:
        return "Please enter a YouTube URL.", None
    print(f"[INPUT] YouTube URL: {url}")
    audio = None
    try:
        audio = fetch_youtube_audio(url)
        text = transcribe(audio)
        return text, None
    except Exception as e:
        print(f"[ERROR] {e}")
        return f"Processing error: {str(e)}", None
    finally:
        if audio and os.path.exists(audio):
            remove_files(audio)

def handle_video_upload(path):
    print(f"[INPUT] Uploaded video: {path}")
    try:
        text = transcribe(path)
        return text, None
    except Exception as e:
        print(f"[ERROR] {e}")
        return f"Processing error: {str(e)}", None

# --- Gradio UI ---
print("[INFO] Launching Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎥 Video Transcription & Smart Summary
        
        Upload a video or paste a YouTube link to get a transcription and an AI-generated summary. Powered by Whisper and LLMs. 
        
        **Author:** [Hit Kalariya](https://github.com/hitkalariya) | [LinkedIn](https://www.linkedin.com/in/hitkalariya/) | [Email](mailto:hitkalariya88@gmail.com)
        """
    )
    with gr.Tabs():
        with gr.TabItem("📤 Video Upload"):
            video_input = gr.Video(label="Upload or drag your video here")
            video_button = gr.Button("Process Video", variant="primary")
        with gr.TabItem("🔗 YouTube Link"):
            url_input = gr.Textbox(label="Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            url_button = gr.Button("Process URL", variant="primary")
    with gr.Row():
        with gr.Column():
            transcription_output = gr.Textbox(label="Transcription", lines=10, show_copy_button=True)
        with gr.Column():
            summary_output = gr.Textbox(label="Summary", lines=10, show_copy_button=True)
    summary_button = gr.Button("Generate Summary", variant="secondary")
    gr.Markdown(
        """
        ### Instructions
        1. Upload a video or paste a YouTube link.
        2. Click 'Process' to get the transcription.
        3. Click 'Generate Summary' for a concise summary.
        
        *Processing time depends on video length.*
        """
    )
    def process_video_and_update(video):
        if video is None:
            return "No video uploaded.", "Please upload a video."
        text, _ = handle_video_upload(video)
        return text or "Transcription error", ""
    video_button.click(process_video_and_update, inputs=[video_input], outputs=[transcription_output, summary_output])
    url_button.click(handle_youtube, inputs=[url_input], outputs=[transcription_output, summary_output])
    summary_button.click(summarize, inputs=[transcription_output], outputs=[summary_output])
demo.launch()
    )
