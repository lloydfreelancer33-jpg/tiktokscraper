import os, json, base64, requests, subprocess, uuid, logging, glob, time, threading, tempfile, shutil
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageStat, ImageOps # Using Pillow instead of CV2
from openai import OpenAI
from waitress import serve
from dotenv import load_dotenv

load_dotenv()
TEMP_DIR = tempfile.gettempdir()

app = Flask(__name__)
CORS(app)

conf = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "SUPABASE_URL": os.environ.get("SUPABASE_URL"),
    "SUPABASE_KEY": os.environ.get("SUPABASE_KEY")
}
client = OpenAI(api_key=conf["OPENAI_API_KEY"])

# --- UTILS ---

def get_ffmpeg_cmd():
    """Detects environment and returns the usable ffmpeg path."""
    # 1. Check for local folder (Windows testing)
    local_path = os.path.join(os.getcwd(), "ffmpeg", "bin", "ffmpeg.exe")
    if os.path.exists(local_path):
        return local_path
    
    # 2. Check if it's in the system PATH (Windows/Linux/Docker)
    import shutil
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path
        
    return "ffmpeg" # Final fallback

def get_blur_score(image_path):
    """
    Lightweight blur detection using Pillow.
    Calculates the standard deviation of the image. 
    Higher = Sharper.
    """
    with Image.open(image_path) as img:
        img = img.convert('L') # Convert to grayscale
        stat = ImageStat.Stat(img)
        return stat.stddev[0] # The standard deviation of pixel values

def extract_best_frames(media_path, output_dir, is_video=True, fps=2):
    """Passes images through; Extracts & filters video frames."""
    if not is_video:
        # It's an image: Copy it and give it a dummy high score
        dest = os.path.join(output_dir, "raw_frame_0001.jpg")
        shutil.copy(media_path, dest)
        return [{"path": dest, "score": 100}]

    ffmpeg_cmd = get_ffmpeg_cmd()
    cmd = [
        ffmpeg_cmd, "-i", media_path,
        "-vf", f"fps={fps}",
        os.path.join(output_dir, "raw_frame_%04d.jpg")
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    all_raw_frames = glob.glob(os.path.join(output_dir, "raw_frame_*.jpg"))
    scored_frames = []
    for path in all_raw_frames:
        score = get_blur_score(path)
        # We only keep frames that aren't pure black or blurry
        if score > 22: 
            scored_frames.append({"path": path, "score": score})
    
    scored_frames.sort(key=lambda x: x['score'], reverse=True)
    return scored_frames

# --- AI LOGIC ---

def get_unique_products(frames_data):
    """Identifies distinct products and selects the best 'Master Frame' for each."""
    # We label each frame so the AI can refer to them as f_0, f_1, etc.
    payload_imgs = []
    for i, f in enumerate(frames_data[:10]):
        payload_imgs.append({
            "type": "text", 
            "text": f"Frame Index: f_{i}"
        })
        payload_imgs.append({
            "type": "image_url", 
            "image_url": {"url": f["b64"], "detail": "low"}
        })
    
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
    
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": payload_imgs}]
    )
    return json.loads(res.choices[0].message.content).get("products", [])

# --- ENDPOINTS ---

@app.route('/process_media', methods=['POST'])
def process_media():
    # 1. Setup IDs and Media Retrieval
    raw_ad_id = request.form.get('ad_id')
    # If no ad_id, it's a TikTok/generic video; generate a unique trackable ID
    source_type = "AD" if raw_ad_id else "TIKTOK"
    final_id = raw_ad_id if raw_ad_id else f"tt_{uuid.uuid4().hex[:8]}"
    
    media_file = request.files.get('media')
    if not media_file:
        return jsonify({"error": "No media provided"}), 400

    # 2. Setup Temporary Workspace
    run_dir = os.path.join(TEMP_DIR, f"proc_{final_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    try:
        filename = media_file.filename.lower()
        is_video = any(filename.endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv'])
        
        media_path = os.path.join(run_dir, f"input_{final_id}")
        media_file.save(media_path)
        
        # 3. Extraction (Pass-through for images, FFMPEG for videos)
        best_frames = extract_best_frames(media_path, run_dir, is_video=is_video, fps=2)
        
        # 4. Prepare Frames for AI Analysis
        processed_frames = []
        for f in best_frames[:10]: # Analyze up to 10 sharp frames
            with open(f['path'], "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode('utf-8')
                processed_frames.append({
                    "b64": f"data:image/jpeg;base64,{b64_data}",
                    "score": f['score']
                })

        if not processed_frames:
            return jsonify({"error": "No clear frames could be extracted"}), 422

        # 5. Get AI Analysis (Grouping products and picking master frames)
        # This now returns a list of products with 'master_frame_index' like "f_0"
        detected_products = get_unique_products(processed_frames)

        # 6. Intelligent Forwarding to Dino Worker
        dino_worker_url = os.environ.get("DINO_WORKER_URL") 
        indexed_count = 0
        
        if dino_worker_url and detected_products:
            # We only send the 'Master Frame' for each unique product identified
            for product in detected_products:
                try:
                    # Extract index from string like "f_2" -> 2
                    m_idx = int(product.get('master_frame_index', 'f_0').split('_')[1])
                    
                    # Safety check on index
                    if m_idx < len(processed_frames):
                        master_b64 = processed_frames[m_idx]['b64']
                        
                        requests.post(
                            f"{dino_worker_url}/index-video-frame",
                            json={
                                "video_id": final_id,
                                "image_url": master_b64,
                                "video_setting": f"Source: {source_type} | Product: {product.get('name')}"
                            },
                            timeout=7
                        )
                        indexed_count += 1
                except Exception as e:
                    print(f"Dino Forwarding failed for {final_id} index {m_idx}: {e}")

        # 7. Final Response
        return jsonify({
            "status": "SUCCESS",
            "id": final_id,
            "source": source_type,
            "media_type": "video" if is_video else "image",
            "detected_products": detected_products,
            "master_frames_indexed": indexed_count
        })

    except Exception as e:
        print(f"Processing Error: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 500

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8080)
