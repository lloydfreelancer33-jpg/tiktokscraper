import os, json, base64, requests, subprocess, uuid, logging, glob, time, threading, tempfile, shutil, collections
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageStat
from openai import OpenAI
from waitress import serve
from dotenv import load_dotenv

load_dotenv()

TEMP_DIR = "/tmp/temp_processing"
os.makedirs(TEMP_DIR, exist_ok=True)
app = Flask(__name__)
CORS(app)

conf = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "SUPABASE_URL": os.environ.get("SUPABASE_URL"),
    "SUPABASE_KEY": os.environ.get("SUPABASE_KEY")
}
client = OpenAI(api_key=conf["OPENAI_API_KEY"])

# --- IN-MEMORY LOGGER FOR FRONTEND ---
server_logs = collections.deque(maxlen=100) # Store last 100 logs

def log_msg(msg, level="INFO"):
    """Saves logs to memory so the HTML frontend can fetch them."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {level}: {msg}"
    server_logs.append(formatted_msg)
    print(formatted_msg) # Also print to standard terminal

@app.route('/logs', methods=['GET'])
def get_logs():
    """Endpoint for test.html to poll logs."""
    return jsonify(list(server_logs))


# --- UTILS ---

def get_ffmpeg_cmd():
    local_path = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.exists(local_path):
        return local_path
    
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path
        
    return "ffmpeg"

def get_blur_score(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('L')
            stat = ImageStat.Stat(img)
            return stat.stddev[0]
    except Exception as e:
        log_msg(f"Failed to score {image_path}: {e}", "WARNING")
        return 0

def extract_best_frames(media_path, output_dir, is_video=True, fps=2):
    log_msg(f"Extracting frames. is_video={is_video}")
    if not is_video:
        dest = os.path.join(output_dir, "raw_frame_0001.jpg")
        shutil.copy(media_path, dest)
        return [{"path": dest, "score": 100}]

    ffmpeg_cmd = get_ffmpeg_cmd()
    cmd = [
        ffmpeg_cmd, "-i", media_path,
        "-vf", f"fps={fps}",
        os.path.join(output_dir, "raw_frame_%04d.jpg")
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        log_msg(f"FFmpeg error: {e.stderr.decode()}", "ERROR")
        raise
    
    all_raw_frames = glob.glob(os.path.join(output_dir, "raw_frame_*.jpg"))
    scored_frames = []
    for path in all_raw_frames:
        score = get_blur_score(path)
        if score > 22: 
            scored_frames.append({"path": path, "score": score})
    
    scored_frames.sort(key=lambda x: x['score'], reverse=True)
    log_msg(f"Extracted {len(scored_frames)} clear frames.")
    return scored_frames

# --- AI LOGIC ---

def get_unique_products(frames_data):
    log_msg("Sending frames to AI for analysis...")
    payload_imgs = []
    for i, f in enumerate(frames_data[:10]):
        payload_imgs.append({"type": "text", "text": f"Frame Index: f_{i}"})
        payload_imgs.append({"type": "image_url", "image_url": {"url": f["b64"], "detail": "low"}})
    
    prompt = """
    You are a visual inventory auditor. Analyze these labeled frames (f_0, f_1, etc.):
    1. Identify all DISTINCT products.
    2. For each product, list which frames it appears in.
    3. Select the ONE 'master_frame_index' that shows the product most clearly.
    
    Return JSON format:
    {
      "products": [
        {
          "name": "Product Name",
          "color": "Color",
          "appearing_in": ["f_0", "f_1"],
          "master_frame_index": "f_0",
          "reason": "Front view, best lighting"
        }
      ]
    }
    """
    
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": payload_imgs}]
        )
        products = json.loads(res.choices[0].message.content).get("products", [])
        log_msg(f"AI identified {len(products)} products.")
        return products
    except Exception as e:
        log_msg(f"OpenAI API Error: {e}", "ERROR")
        raise

# --- ENDPOINTS ---

@app.route('/process_media', methods=['POST'])
def process_media():
    log_msg("--- NEW REQUEST RECEIVED ---")
    raw_ad_id = request.form.get('ad_id')
    source_type = "AD" if raw_ad_id else "TIKTOK"
    final_id = raw_ad_id if raw_ad_id else f"tt_{uuid.uuid4().hex[:8]}"
    
    media_file = request.files.get('media')
    if not media_file:
        log_msg("No media provided in request.", "ERROR")
        return jsonify({"error": "No media provided"}), 400

    run_dir = os.path.join(TEMP_DIR, f"proc_{final_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    try:
        filename = media_file.filename.lower() if media_file.filename else "unknown.mp4"
        is_video = any(filename.endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv'])
        
        media_path = os.path.join(run_dir, f"input_{final_id}")
        media_file.save(media_path)
        log_msg(f"Saved upload to {media_path}")
        
        best_frames = extract_best_frames(media_path, run_dir, is_video=is_video, fps=2)
        
        processed_frames = []
        for f in best_frames[:10]: 
            with open(f['path'], "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode('utf-8')
                processed_frames.append({
                    "b64": f"data:image/jpeg;base64,{b64_data}",
                    "score": f['score']
                })

        if not processed_frames:
            log_msg("No clear frames could be extracted.", "ERROR")
            return jsonify({"error": "No clear frames could be extracted"}), 422

        detected_products = get_unique_products(processed_frames)

        dino_worker_url = os.environ.get("DINO_WORKER_URL") 
        indexed_count = 0
        
        if dino_worker_url and detected_products:
            log_msg(f"Forwarding {len(detected_products)} master frames to Dino...")
            for product in detected_products:
                try:
                    m_idx_str = product.get('master_frame_index', 'f_0')
                    m_idx = int(m_idx_str.split('_')[1])
                    
                    if m_idx < len(processed_frames):
                        master_b64 = processed_frames[m_idx]['b64']
                        requests.post(
                            f"{dino_worker_url}/index-video-frame",
                            json={
                                "video_id": final_id,
                                "image_url": master_b64,
                                "video_setting": f"Source: {source_type} | Product: {product.get('name')}"
                            },
                            timeout=30
                        )
                        indexed_count += 1
                except Exception as e:
                    log_msg(f"Dino Forwarding failed for {final_id} index {m_idx_str}: {e}", "ERROR")
        else:
            if not dino_worker_url:
                log_msg("DINO_WORKER_URL not set. Skipping vector indexing.", "WARNING")

        log_msg("Processing completed successfully.")
        return jsonify({
            "status": "SUCCESS",
            "id": final_id,
            "source": source_type,
            "media_type": "video" if is_video else "image",
            "detected_products": detected_products,
            "master_frames_indexed": indexed_count
        })

    except Exception as e:
        log_msg(f"Fatal Processing Error: {e}", "ERROR")
        return jsonify({"status": "ERROR", "message": str(e)}), 500

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

if __name__ == "__main__":
    log_msg("Server starting on port 8080...")
    serve(app, host='0.0.0.0', port=8080)
