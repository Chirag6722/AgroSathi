from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from flask_babel import Babel, gettext as _
import os
import hashlib
import joblib
import pandas as pd
import traceback
from werkzeug.utils import secure_filename
from PIL import Image
import json
import torch
import requests
import time
import pathlib
from flask import current_app
from datetime import datetime, timedelta
import torch.nn.functional as F
from torchvision import transforms, models
from flask import session, redirect, request, url_for
from dotenv import load_dotenv
load_dotenv()      # loads .env into environment variables




# ---------------- App & Babel config ----------------
app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-random-key"

# Configure Babel
babel = Babel(app)

# Languages you want available in the UI (code -> display name)
LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी",
    "kn": "ಕನ್ನಡ"
}

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Paths for crop advisory model (kept for your other endpoints)
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_model.joblib")

# Disease classification model files (PyTorch)
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.pth")
DISEASE_LABELS_PATH = os.path.join(BASE_DIR, "models", "disease_labels.json")

# Global cache
_DISEASE_MODEL = None
_DISEASE_LABELS = None
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- Utilities ----------------
def allowed_file(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXT

def load_disease_labels():
    global _DISEASE_LABELS
    if _DISEASE_LABELS is not None:
        return _DISEASE_LABELS
    if not os.path.exists(DISEASE_LABELS_PATH):
        print(f"[disease] labels file not found: {DISEASE_LABELS_PATH}")
        return None
    with open(DISEASE_LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    if isinstance(labels, dict):
        try:
            ordered = [labels[str(i)] for i in range(len(labels))]
            _DISEASE_LABELS = ordered
            return _DISEASE_LABELS
        except Exception:
            _DISEASE_LABELS = list(labels.values())
            return _DISEASE_LABELS
    elif isinstance(labels, list):
        _DISEASE_LABELS = labels
        return _DISEASE_LABELS
    else:
        return None

def build_resnet(num_classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def load_disease_model():
    """
    Loads PyTorch model and labels. Returns (model, labels) or (None, None) on failure.
    """
    global _DISEASE_MODEL, _DISEASE_LABELS

    if _DISEASE_MODEL is not None and _DISEASE_LABELS is not None:
        return _DISEASE_MODEL, _DISEASE_LABELS

    labels = load_disease_labels()
    if labels is None:
        print("[disease] labels missing; cannot load model.")
        return None, None

    num_classes = len(labels)
    if not os.path.exists(DISEASE_MODEL_PATH):
        print(f"[disease] model file not found: {DISEASE_MODEL_PATH}")
        return None, None

    try:
        model = build_resnet(num_classes)
        state = torch.load(DISEASE_MODEL_PATH, map_location=_DEVICE)
        if isinstance(state, dict) and 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state
        new_state = {}
        for k, v in state_dict.items():
            nk = k.replace('module.', '')
            new_state[nk] = v
        model.load_state_dict(new_state)
        model.to(_DEVICE)
        model.eval()
        _DISEASE_MODEL = model
        _DISEASE_LABELS = labels
        print(f"[disease] Loaded model ({DISEASE_MODEL_PATH}) with {num_classes} classes on {_DEVICE}")
        return model, labels
    except Exception as e:
        print("[disease] Failed to load model:", e)
        traceback.print_exc()
        return None, None

# Preprocessing transform for input images (same used in training script)
IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Babel locale selector (cross-version safe) ----------------
# we assign the function to babel.locale_selector_func so it works regardless of decorator availability
def get_locale():
    # First check session for chosen language
    lang = session.get('lang')
    if lang in LANGUAGES:
        return lang
    # Fallback to best match from request headers
    best = request.accept_languages.best_match(list(LANGUAGES.keys()))
    return best or 'en'

# assign to babel
babel.locale_selector_func = get_locale

# Make current language and languages available in all templates
@app.context_processor
def inject_conf_vars():
    return {
        "CURRENT_LANG": get_locale(),
        "LANGUAGES": LANGUAGES
    }


# ---------- Weather endpoint using /weather + /forecast (free plan) ----------
from datetime import datetime, timedelta
import requests, time
from collections import Counter

_OWM_CACHE = {}
_OWM_CACHE_TTL = 60 * 10  # 10 minutes

def _cache_get(key):
    e = _OWM_CACHE.get(key)
    if not e:
        return None
    ts, data = e
    if time.time() - ts > _OWM_CACHE_TTL:
        _OWM_CACHE.pop(key, None)
        return None
    return data

def _cache_set(key, data):
    _OWM_CACHE[key] = (time.time(), data)

@app.route('/weather')
def weather_page():
    return render_template('weather.html')

@app.route('/api/weather', methods=['POST'])
def weather_api():
    """
    POST JSON:
      - { "city": "Bengaluru" }  OR
      - { "lat": 12.97, "lon": 77.59 }
    Returns aggregated 5-day forecast + current conditions.
    """
    try:
        payload = request.get_json(force=True) or {}
        city = (payload.get('city') or '').strip()
        lat = payload.get('lat')
        lon = payload.get('lon')

        # Read API key
        OWM_KEY = os.environ.get('OWM_KEY')
        if not OWM_KEY:
            return jsonify({"error": "OpenWeather API key not configured (OWM_KEY)."}), 500

        # If city provided but no lat/lon, geocode (cached)
        if city and (lat is None or lon is None):
            geo_key = f"geo:{city.lower()}"
            geo_cached = _cache_get(geo_key)
            if geo_cached:
                lat, lon = geo_cached['lat'], geo_cached['lon']
            else:
                geourl = "http://api.openweathermap.org/geo/1.0/direct"
                gr = requests.get(geourl, params={"q": city, "limit": 1, "appid": OWM_KEY}, timeout=8)
                if gr.status_code != 200:
                    return jsonify({"error": "Geocoding failed", "details": gr.text}), 502
                gdata = gr.json()
                if not isinstance(gdata, list) or len(gdata) == 0:
                    return jsonify({"error": "City not found (geocoding)"}), 404
                lat, lon = gdata[0]['lat'], gdata[0]['lon']
                _cache_set(geo_key, {"lat": lat, "lon": lon})

        if lat is None or lon is None:
            return jsonify({"error": "Provide 'city' or valid 'lat' and 'lon'"}), 400

        # cache by coords + date
        cache_key = f"owm_wf:{float(lat):.4f}:{float(lon):.4f}:{datetime.utcnow().date().isoformat()}"
        cached = _cache_get(cache_key)
        if cached:
            return jsonify(cached)

        # current weather (free endpoint)
        cur_url = "https://api.openweathermap.org/data/2.5/weather"
        rcur = requests.get(cur_url, params={"lat": lat, "lon": lon, "units": "metric", "appid": OWM_KEY}, timeout=8)
        if rcur.status_code == 401:
            return jsonify({"error": "OpenWeather: invalid API key (weather endpoint)."}), 502
        rcur.raise_for_status()
        cur = rcur.json()

        # 5-day forecast (3-hour steps)
        f_url = "https://api.openweathermap.org/data/2.5/forecast"
        rf = requests.get(f_url, params={"lat": lat, "lon": lon, "units": "metric", "appid": OWM_KEY}, timeout=9)
        if rf.status_code == 401:
            return jsonify({"error": "OpenWeather: invalid API key (forecast endpoint)."}), 502
        rf.raise_for_status()
        fdata = rf.json()

        # aggregate forecast into daily bins (today + next 4 days)
        today = datetime.utcfromtimestamp(cur.get("dt", time.time())).date()
        bins = {}

        def add_to_bin(dtt, item):
            day = dtt.date().isoformat()
            if day not in bins:
                bins[day] = {"temps_min": [], "temps_max": [], "pops": [], "precip_mm": 0.0, "summaries": []}
            b = bins[day]
            main = item.get("main", {})
            if "temp_min" in main and main.get("temp_min") is not None: b["temps_min"].append(main.get("temp_min"))
            if "temp_max" in main and main.get("temp_max") is not None: b["temps_max"].append(main.get("temp_max"))
            pop = item.get("pop", 0.0)
            b["pops"].append(pop)
            rain = 0.0
            if item.get("rain") and isinstance(item.get("rain"), dict):
                rain = float(item["rain"].get("3h", 0.0))
            if item.get("snow") and isinstance(item.get("snow"), dict):
                rain += float(item["snow"].get("3h", 0.0))
            b["precip_mm"] += rain
            w = item.get("weather") or []
            if w:
                b["summaries"].append(w[0].get("main",""))

        # include current moment as a point in the day
        try:
            add_to_bin(datetime.utcfromtimestamp(cur.get("dt", time.time())), {"main": {"temp_min": cur.get("main",{}).get("temp"), "temp_max": cur.get("main",{}).get("temp")}, "pop": 0.0, "rain": {}, "weather": cur.get("weather",[])})
        except:
            pass

        for item in fdata.get("list", []):
            dt = datetime.utcfromtimestamp(item.get("dt"))
            add_to_bin(dt, item)

        out_forecast = []
        sorted_days = sorted(bins.keys())
        for d in sorted_days[:5]:
            b = bins[d]
            tmin = round(min(b["temps_min"]) if b["temps_min"] else None, 1) if b["temps_min"] else None
            tmax = round(max(b["temps_max"]) if b["temps_max"] else None, 1) if b["temps_max"] else None
            pop = round(max(b["pops"]) if b["pops"] else 0.0, 2)
            precip = round(b["precip_mm"], 1)
            summary = None
            if b["summaries"]:
                summary = Counter(b["summaries"]).most_common(1)[0][0]
            out_forecast.append({
                "date": d,
                "temp_min_c": tmin,
                "temp_max_c": tmax,
                "rain_probability": pop,
                "precip_mm": precip,
                "summary": summary
            })

        result = {
            "source": "openweather:weather+forecast",
            "city": city or cur.get("name"),
            "lat": float(lat),
            "lon": float(lon),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "current": {
                "temp_c": round(cur.get("main",{}).get("temp"),1) if cur.get("main") else None,
                "weather": (cur.get("weather") or [{}])[0].get("description","")
            },
            "forecast": out_forecast
        }

        _cache_set(cache_key, result)
        return jsonify(result)

    except requests.exceptions.RequestException as e:
        print("OpenWeather request failed:", e)
        return jsonify({"error": "OpenWeather request failed", "details": str(e)}), 502
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Server error", "details": str(e)}), 500



# Route to change language (called by the dropdown)
@app.route('/set_language/<lang_code>')
def set_language(lang_code):
    if lang_code not in LANGUAGES:
        return redirect(request.referrer or url_for('home'))
    session['lang'] = lang_code
    # redirect back
    return redirect(request.referrer or url_for('home'))


# ---------------- Pages ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/smart-advisory')
def smart_advisory():
    advisory_content = {
        "title": _("Smart Crop Advisory"),
        "subtitle": _("Get personalized recommendations for planting, irrigation, and harvesting based on AI analysis of weather, soil, and crop data."),
        "items": [
            {"heading": _("Optimal planting schedules"), "detail": _("AI-driven schedules optimized by weather and soil conditions.")},
            {"heading": _("Water management"), "detail": _("Irrigation timing and volumes based on soil moisture and forecasts.")},
            {"heading": _("Harvest timing"), "detail": _("Recommendations for best harvest window to maximize yield & quality.")}
        ]
    }
    return render_template('smart_advisory.html', advisory=advisory_content)

@app.route('/disease-detection')
def disease_detection_page():
    return render_template('disease_detection.html')

# ---------------- Soil Analysis Page ----------------
@app.route('/soil-analysis')
def soil_analysis_page():
    """
    Renders the Soil Analysis page when the soil card is clicked.
    Create templates/soil_analysis.html (see step 2).
    """
    return render_template('soil_analysis.html')

# ---------- Government Schemes (Farmers) ----------
import pathlib

SCHEMES_FILE = os.path.join(BASE_DIR, "data", "schemes.json")
# ensure data folder exists
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

def load_schemes():
    """Load schemes from local JSON file. Returns list of dicts."""
    if not os.path.exists(SCHEMES_FILE):
        # If file missing, create a small sample set
        sample = [
            {
                "id": "pmfby",
                "title": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
                "summary": "Crop insurance to support farmers with financial compensation for crop loss due to natural calamities.",
                "eligibility": "All farmers growing notified crops",
                "benefits": "Insurance coverage against yield loss and post-harvest losses.",
                "apply_url": "https://pmfby.gov.in/",
                "source": "Govt. of India",
                "last_updated": "2024-01-01"
            },
            {
                "id": "pmkisan",
                "title": "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
                "summary": "Direct income support of ₹6,000/year to small and marginal farmer families.",
                "eligibility": "Small & marginal farmer families with landholding",
                "benefits": "Quarterly direct cash transfers.",
                "apply_url": "https://pmkisan.gov.in/",
                "source": "Govt. of India",
                "last_updated": "2024-01-01"
            },
            {
                "id": "soilhealthcard",
                "title": "Soil Health Card Scheme",
                "summary": "Provides soil testing and recommendations to farmers to improve soil health.",
                "eligibility": "All farmers",
                "benefits": "Soil test report with fertilizer recommendations.",
                "apply_url": "https://soilhealth.dac.gov.in/",
                "source": "Govt. of India",
                "last_updated": "2024-01-01"
            }
        ]
        with open(SCHEMES_FILE, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        return sample

    try:
        with open(SCHEMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # expect list of objects
            return data if isinstance(data, list) else []
    except Exception as e:
        print("[schemes] failed to load:", e)
        return []


    try:
        schemes = load_schemes()
        q = request.args.get('q', '').strip().lower()
        source_filter = request.args.get('source', '').strip().lower()
        if q:
            schemes = [
                s for s in schemes
                if q in (s.get('title','').lower() + ' ' + s.get('summary','').lower() + ' ' + s.get('eligibility','').lower())
            ]
        if source_filter:
            schemes = [s for s in schemes if source_filter == s.get('source','').lower()]
        return jsonify({"count": len(schemes), "schemes": schemes})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to load schemes", "details": str(e)}), 500


# Serve uploaded images
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ---------------- Disease Detection API ----------------
@app.route('/api/detect-disease', methods=['POST'])
def detect_disease_api():
    try:
        if 'image' not in request.files:
            return jsonify({"error": _("No file part 'image'")}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": _("No selected file")}), 400

        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            return jsonify({"error": _("Unsupported file extension")}), 400

        save_name = f"{hashlib.sha1((filename + str(os.urandom(8))).encode()).hexdigest()}.{filename.rsplit('.',1)[-1].lower()}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(save_path)

        # Validate image
        try:
            img = Image.open(save_path).convert('RGB')
        except Exception:
            try: os.remove(save_path)
            except: pass
            return jsonify({"error": _("Uploaded file is not a valid image")}), 400

        # Try to load real model
        model, labels = load_disease_model()

        if model is None or labels is None:
            # deterministic demo fallback
            with open(save_path, 'rb') as f:
                b = f.read()
            h = int(hashlib.sha256(b).hexdigest()[:16], 16)
            diseases = [
                "Healthy",
                "Late Blight",
                "Powdery Mildew",
                "Leaf Spot",
                "Rust",
                "Bacterial Spot",
                "Downy Mildew"
            ]
            idx = h % len(diseases)
            picks = []
            for i in range(3):
                di = (idx + i) % len(diseases)
                prob = 0.5 + 0.5 * (((h >> (i*7)) % 100) / 100.0)
                picks.append((diseases[di], round(min(prob, 0.999), 3)))

            predictions = [{"disease": d, "probability": p} for d, p in picks]
            top_disease = predictions[0]['disease']
            recommendations = detailed_treatment_map().get(top_disease, [_("Follow general integrated pest management (IPM) steps.")])
            return jsonify({"predictions": predictions, "top_solution": recommendations, "image_url": f"/uploads/{save_name}"})

        # Real model inference
        try:
            inp = IMG_TRANSFORMS(img).unsqueeze(0).to(_DEVICE)
            with torch.no_grad():
                out = model(inp)
                probs = F.softmax(out, dim=1).cpu().numpy()[0]
            pairs = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)[:3]
            predictions = [{"disease": labels[idx], "probability": float(round(float(p), 4))} for idx, p in pairs]
            top_disease = predictions[0]['disease']
            recommendations = detailed_treatment_map().get(top_disease, [_("Follow integrated pest management best practices.")])
            return jsonify({"predictions": predictions, "top_solution": recommendations, "image_url": f"/uploads/{save_name}"})
        except Exception as e:
            print("[disease] inference error:", e)
            traceback.print_exc()
            # fallback deterministic with labels if available
            with open(save_path, 'rb') as f:
                b = f.read()
            h = int(hashlib.sha256(b).hexdigest()[:16], 16)
            diseases = labels if labels else ["Healthy","Late Blight","Powdery Mildew","Leaf Spot","Rust","Bacterial Spot","Downy Mildew"]
            idx = h % len(diseases)
            picks = []
            for i in range(3):
                di = (idx + i) % len(diseases)
                prob = 0.5 + 0.5 * (((h >> (i*7)) % 100) / 100.0)
                picks.append((diseases[di], round(min(prob, 0.999), 3)))
            predictions = [{"disease": d, "probability": p} for d, p in picks]
            top_disease = predictions[0]['disease']
            recommendations = detailed_treatment_map().get(top_disease, [_("Follow general integrated pest management (IPM) steps.")])
            return jsonify({"predictions": predictions, "top_solution": recommendations, "image_url": f"/uploads/{save_name}"})

    except Exception as outer:
        print("Unhandled exception in /api/detect-disease:", outer)
        traceback.print_exc()
        return jsonify({"error": _("Server error"), "details": str(outer)}), 500
    


# ---------------- Detailed treatment mapping ----------------
def detailed_treatment_map():
    return {
        "Healthy": [
            _("No treatment required. Continue regular crop monitoring."),
            _("Maintain balanced fertilization and ensure proper irrigation.")
        ],
        "Late Blight": [
            _("Immediately remove and destroy infected plants and debris."),
            _("Avoid overhead irrigation to reduce leaf wetness duration."),
            _("Apply fungicides containing Mancozeb (0.25%), Metalaxyl-M + Mancozeb (Ridomil Gold), or Cymoxanil + Mancozeb (Curzate) at 7-10 day intervals during cool, humid weather.")
        ],
        "Powdery Mildew": [
            _("Prune infected leaves and ensure proper air movement in canopy."),
            _("Apply Sulfur 80% WP (2 g/L) or Trifloxystrobin + Tebuconazole (Nativo 75 WG, 0.1%)."),
            _("Repeat sprays every 10-12 days depending on severity.")
        ],
        "Leaf Spot": [
            _("Collect and burn infected leaves; avoid splashing irrigation."),
            _("Apply Chlorothalonil (2 g/L) or Carbendazim (1 g/L) or Copper oxychloride (3 g/L) at 10-day intervals."),
            _("Rotate crops and use disease-free seed.")
        ],
        "Rust": [
            _("Remove rusted leaves and improve ventilation."),
            _("Spray Hexaconazole 5% EC (1 ml/L) or Propiconazole 25% EC (1 ml/L) or Mancozeb (2 g/L)."),
            _("Repeat after 10-14 days if symptoms persist.")
        ],
        "Bacterial Spot": [
            _("Remove infected leaves and fruits; avoid working in wet fields."),
            _("Use certified clean seed and resistant varieties if available."),
            _("Spray Copper hydroxide (Kocide 101, 2.5 g/L) or Copper oxychloride (3 g/L) mixed with Streptomycin sulfate (0.1 g/L).")
        ],
        "Downy Mildew": [
            _("Destroy infected residues and ensure proper drainage."),
            _("Apply Metalaxyl + Mancozeb (Ridomil Gold MZ, 2.5 g/L) or Dimethomorph (Acrobat, 1.5 g/L) at 7-10 day intervals."),
            _("Rotate fungicides to prevent resistance buildup.")
        ]
    }

# ---------------- Crop advisory endpoints (kept from your existing app) ----------------
_CROP_MODEL_CACHE = None
def ensure_model():
    global _CROP_MODEL_CACHE
    if _CROP_MODEL_CACHE is not None:
        return _CROP_MODEL_CACHE
    if not os.path.exists(MODEL_PATH):
        print("Crop model not found. (If you want, run your demo training to create it.)")
        return None
    try:
        _CROP_MODEL_CACHE = joblib.load(MODEL_PATH)
        return _CROP_MODEL_CACHE
    except Exception as e:
        print("Failed to load crop model:", e)
        return None

@app.route('/api/crop-advisory', methods=['POST'])
def crop_advisory_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": _("Missing JSON body")}), 400

        required = ['soil', 'temp', 'ph', 'season', 'rainfall']
        for r in required:
            if r not in data:
                return jsonify({"error": _("Missing required field: ") + r}), 400

        model = ensure_model()
        if model is None:
            return jsonify({
                "predictions": [{"crop":"Wheat","probability":0.8}],
                "note": _("Crop model not loaded; this is a sample response.")
            })

        X = pd.DataFrame([{
            'soil': data['soil'],
            'season': data['season'],
            'temp': float(data['temp']),
            'ph': float(data['ph']),
            'rainfall': float(data['rainfall'])
        }])

        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                classes = getattr(model, "classes_", None)
                if classes is None and hasattr(model, "named_steps"):
                    for step in reversed(list(model.named_steps.values())):
                        if hasattr(step, "classes_"):
                            classes = step.classes_
                            break
                if classes is None:
                    classes = ["Wheat"]
                pairs = sorted(zip(classes, proba[0]), key=lambda x: x[1], reverse=True)
                return jsonify({"predictions":[{"crop":str(c),"probability":round(float(p),3)} for c,p in pairs[:3]]})
        except Exception:
            pred = model.predict(X)
            return jsonify({"predictions":[{"crop":str(pred[0]),"probability":1.0}]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":_("Server error"),"details":str(e)}), 500


# ---------------- Soil Analysis API ----------------
def _convert_possible_ppm_to_kg_per_ha(value):
    """
    Heuristic conversion: if the value is small (looks like ppm), convert using factor ~2:
      1 ppm ≈ 2 kg/ha for 0-15 cm soil layer (approx, depends on bulk density).
    If value already large (e.g. >50) we assume it's kg/ha and return unchanged.
    """
    try:
        v = float(value)
    except Exception:
        return None
    if v <= 50:   # heuristic threshold: treat as ppm if small
        return v * 2.0
    return v

def _compute_ph_advice(pH):
    pH = float(pH)
    if pH < 5.5:
        return {
            "status": "Acidic",
            "advice": "Apply agricultural lime (calcitic or dolomitic) to raise pH. Start with 500-2000 kg/ha depending on soil buffer capacity and crop; consult local agronomist for exact liming rate."
        }
    elif pH < 6.5:
        return {
            "status": "Slightly acidic",
            "advice": "Consider light liming and organic matter additions (compost). Many crops prefer pH 6.0-7.0."
        }
    elif pH <= 7.5:
        return {
            "status": "Neutral to Slightly alkaline",
            "advice": "pH is good for most crops. Maintain organic matter, ensure balanced fertilization."
        }
    else:
        return {
            "status": "Alkaline",
            "advice": "Soil is alkaline; consider sulfur or acidifying amendments and use crop varieties tolerant of higher pH. Consult specialist for gypsum/sulfur rates."
        }

@app.route('/api/soil-analysis', methods=['POST'])
def soil_analysis_api():
    """
    Simple rule-based soil analysis endpoint.
    Expects JSON body with: N, P, K, ph
    Optional: moisture, area_ha, units ("ppm" or "kg/ha")
    Returns json with deficits, suggestions, ph_advice, summary, predicted_class, probabilities, etc.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        # required fields
        for field in ['N', 'P', 'K', 'ph']:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # parse inputs
        raw_N = data.get('N')
        raw_P = data.get('P')
        raw_K = data.get('K')
        raw_ph = data.get('ph')
        moisture = data.get('moisture', None)
        area_ha = float(data.get('area_ha', 1.0))
        units = data.get('units', None)  # optional; if 'ppm' we convert; otherwise auto-detect

        # auto-detect / convert to kg/ha
        # if user explicitly provided units='ppm' convert using factor; else heuristically convert small numbers
        if units == 'ppm':
            N_kg_ha = float(raw_N) * 2.0
            P_kg_ha = float(raw_P) * 2.0
            K_kg_ha = float(raw_K) * 2.0
        else:
            N_kg_ha = _convert_possible_ppm_to_kg_per_ha(raw_N)
            P_kg_ha = _convert_possible_ppm_to_kg_per_ha(raw_P)
            K_kg_ha = _convert_possible_ppm_to_kg_per_ha(raw_K)

        ph = float(raw_ph)

        # basic targets (generic crop-agnostic; you can extend per-crop later)
        # These are typical guideline target macronutrient levels (kg/ha) for many crops.
        targets = {
            "N": 100.0,
            "P": 40.0,
            "K": 80.0
        }

        # compute deficits and totals for area
        deficits = {}
        suggestions = []
        ratios = []
        for nut, cur in (('N', N_kg_ha), ('P', P_kg_ha), ('K', K_kg_ha)):
            if cur is None:
                cur = 0.0
            target = targets[nut]
            deficit = max(0.0, round(target - float(cur), 2))
            total_deficit = round(deficit * area_ha, 2)
            deficits[nut] = {
                "current_kg_per_ha": round(float(cur), 2),
                "target_kg_per_ha": round(target, 2),
                "deficit_kg_per_ha": round(deficit, 2),
                "total_deficit_for_area_kg": round(total_deficit, 2)
            }
            # prepare suggestion: apply deficit (or 0) as baseline
            suggest_apply = round(deficit, 2)
            suggestions.append({
                "nutrient": nut,
                "suggest_apply_kg_per_ha": suggest_apply,
                "suggest_apply_total_kg_for_area": round(suggest_apply * area_ha, 2),
                "note": "Apply as per product nutrient analysis (convert to fertilizer product kg)."
            })
            # ratio current/target for fertility classification
            ratios.append( (float(cur) / target) if target > 0 else 1.0 )

        # classification heuristic
        avg_ratio = sum(ratios) / len(ratios)
        if avg_ratio >= 1.0:
            predicted = "High"
            probs = {"High": 0.85, "Medium": 0.10, "Low": 0.05}
        elif avg_ratio >= 0.6:
            predicted = "Medium"
            # make pseudo probabilities proportional to distance from thresholds
            p_med = 0.5 + (avg_ratio - 0.6) * 1.25  # maps 0.6->0.5, 1.0->1.0
            p_high = max(0.05, (avg_ratio - 0.8) * 1.25)
            p_low = max(0.05, 1.0 - p_med - p_high)
            # normalize
            s = p_med + p_high + p_low
            probs = {"High": round(p_high/s, 3), "Medium": round(p_med/s, 3), "Low": round(p_low/s, 3)}
        else:
            predicted = "Low"
            probs = {"High": 0.03, "Medium": 0.25, "Low": 0.72}

        # pH advice
        ph_advice = _compute_ph_advice(ph)

        # build summary
        deficits_present = any(deficits[n]['deficit_kg_per_ha'] > 0 for n in deficits)
        if deficits_present:
            summary = "Deficits found in macronutrients. Apply suggested nutrients and re-test after one cropping season."
        else:
            summary = "Soil macronutrients meet or exceed generic targets. Maintain balanced fertilization & organic matter."

        # return structured JSON
        resp = {
            "source": "Rule-based",
            "predicted_class": predicted,
            "probabilities": probs,
            "input": {
                "N": round(float(N_kg_ha),2),
                "P": round(float(P_kg_ha),2),
                "K": round(float(K_kg_ha),2),
                "ph": round(float(ph),2),
                "moisture": moisture,
                "area_ha": area_ha,
                "units_assumed": units or ("ppm->kg/ha" if float(data.get('N',0)) <= 50 else "kg/ha")
            },
            "targets_used": targets,
            "deficits": deficits,
            "suggestions": suggestions,
            "ph_advice": ph_advice,
            "summary": summary
        }

        return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Server error", "details": str(e)}), 500
    
# --- Schemes (Government schemes for farmers) ---
DATA_DIR = os.path.join(BASE_DIR, "data")
SCHEMES_FILE = os.path.join(DATA_DIR, "schemes_india.json")

@app.route('/schemes')
def schemes_page():
    # Render a page (templates/schemes.html)
    return render_template('schemes.html')

@app.route('/api/schemes', methods=['GET'])
def api_schemes():
    """
    Returns list of schemes from local JSON file.
    Query params (optional):
     - ministry (filter by ministry substring)
     - q (search in title/short_desc)
     - updated_since (YYYY-MM-DD) - filters last_updated >= date
    """
    try:
        if not os.path.exists(SCHEMES_FILE):
            return jsonify({"schemes": [], "note": "No schemes file found; create data/schemes_india.json"}), 200

        import datetime
        with open(SCHEMES_FILE, 'r', encoding='utf-8') as f:
            all_schemes = json.load(f)

        # basic filters
        ministry = (request.args.get('ministry') or "").strip().lower()
        q = (request.args.get('q') or "").strip().lower()
        updated_since = request.args.get('updated_since')

        def keep(s):
            if ministry and ministry not in (s.get('ministry','').lower()):
                return False
            if q and q not in (s.get('title','').lower() + " " + s.get('short_desc','').lower()):
                return False
            if updated_since:
                try:
                    d = datetime.datetime.strptime(s.get('last_updated','1970-01-01')[:10], "%Y-%m-%d").date()
                    ds = datetime.datetime.strptime(updated_since[:10], "%Y-%m-%d").date()
                    if d < ds:
                        return False
                except Exception:
                    pass
            return True

        filtered = [s for s in all_schemes if keep(s)]
        return jsonify({"schemes": filtered})
    except Exception as e:
        current_app.logger.exception("Failed to load schemes")
        return jsonify({"error": "Server error", "details": str(e)}), 500

    
# ---------------- Soil analysis ML endpoint ----------------
SOIL_MODEL_PATH = os.path.join(BASE_DIR, "models", "soil_model.joblib")
_SOIL_MODEL = None

def ensure_soil_model():
    global _SOIL_MODEL
    if _SOIL_MODEL is not None:
        return _SOIL_MODEL
    if not os.path.exists(os.path.join(BASE_DIR, "models")):
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    if not os.path.exists(SOIL_MODEL_PATH):
        print(f"[soil] model not found: {SOIL_MODEL_PATH} -> falling back to rule-based")
        return None
    try:
        _SOIL_MODEL = joblib.load(SOIL_MODEL_PATH)
        print(f"[soil] loaded model from {SOIL_MODEL_PATH}")
        return _SOIL_MODEL
    except Exception as e:
        print("[soil] failed to load model:", e)
        traceback.print_exc()
        return None

def rule_based_prediction(N, P, K, ph, moisture=None):
    thr = {'N': 40.0, 'P': 20.0, 'K': 150.0}
    score = 0.0
    score += min(N / thr['N'], 2.0) * 0.4
    score += min(P / thr['P'], 2.0) * 0.3
    score += min(K / thr['K'], 2.0) * 0.3
    if score >= 1.2:
        pred = "High"
        probs = {"High": 0.85, "Medium": 0.12, "Low": 0.03}
    elif score >= 0.7:
        pred = "Medium"
        probs = {"High": 0.15, "Medium": 0.70, "Low": 0.15}
    else:
        pred = "Low"
        probs = {"High": 0.02, "Medium": 0.18, "Low": 0.80}
    return pred, probs

def compute_deficits_and_suggestions(N, P, K, area_ha=1.0, units="ppm"):
    ppm_to_kg = 2.0
    if units == "kg/ha":
        measured_kg = {'N': float(N), 'P': float(P), 'K': float(K)}
    else:
        measured_kg = {'N': float(N) * ppm_to_kg, 'P': float(P) * ppm_to_kg, 'K': float(K) * ppm_to_kg}

    targets = {'N': 80.0, 'P': 30.0, 'K': 200.0}

    deficits = {}
    suggestions = []
    for nut in ('N','P','K'):
        measured = measured_kg.get(nut, 0.0)
        deficit = max(0.0, round(targets[nut] - measured, 1))
        apply_kg_per_ha = round(deficit, 1)
        total_apply = round(apply_kg_per_ha * float(area_ha), 2)
        note = "Apply to reach target nutrient level; split across applications as needed."
        deficits[nut] = {
            "measured_kg_per_ha": round(measured, 2),
            "target_kg_per_ha": targets[nut],
            "deficit_kg_per_ha": deficit
        }
        suggestions.append({
            "nutrient": nut,
            "suggest_apply_kg_per_ha": apply_kg_per_ha,
            "suggest_apply_total_kg_for_area": total_apply,
            "note": note
        })
    return deficits, suggestions

def ph_advice(ph_val):
    try:
        ph = float(ph_val)
    except:
        return {"status": "", "advice": ""}
    if ph < 5.5:
        return {"status": "Acidic", "advice": "Apply agricultural lime to raise pH; follow local extension rates."}
    elif ph <= 7.5:
        return {"status": "Neutral/Optimal", "advice": "pH is in the good range for many crops."}
    else:
        return {"status": "Alkaline", "advice": "Consider acidifying amendments; consult local guidance."}

@app.route('/api/soil-analysis', methods=['POST'])
def api_soil_analysis():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid JSON", "details": str(e)}), 400

    try:
        N = float(data.get('N', 0.0))
        P = float(data.get('P', 0.0))
        K = float(data.get('K', 0.0))
        ph_val = float(data.get('ph', 0.0))
    except Exception as e:
        return jsonify({"error": "Missing or invalid N/P/K/ph values", "details": str(e)}), 400

    moisture = data.get('moisture', None)
    try:
        moisture = float(moisture) if moisture is not None else None
    except:
        moisture = None

    area_ha = float(data.get('area_ha', 1.0))
    units = data.get('units', 'ppm')

    model = ensure_soil_model()
    if model is not None:
        try:
            X = pd.DataFrame([{'N': N, 'P': P, 'K': K, 'ph': ph_val, 'moisture': (moisture if moisture is not None else 0.0)}])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                classes = list(getattr(model, "classes_", []))
                if not classes:
                    classes = ["Low", "Medium", "High"]
                probs_dict = {str(classes[i]): float(round(float(probs[i]), 4)) for i in range(len(classes))}
                pred_class = max(probs_dict.items(), key=lambda x: x[1])[0]
            else:
                pred_class = str(model.predict(X)[0])
                probs_dict = {pred_class: 1.0}
            source = "AI model"
        except Exception as e:
            print("[soil] model inference failed, falling back to rule-based:", e)
            traceback.print_exc()
            pred_class, probs_dict = rule_based_prediction(N, P, K, ph_val, moisture)
            source = "rule-based"
    else:
        pred_class, probs_dict = rule_based_prediction(N, P, K, ph_val, moisture)
        source = "rule-based"

    deficits, suggestions = compute_deficits_and_suggestions(N, P, K, area_ha=area_ha, units=units)
    phinfo = ph_advice(ph_val)

    resp = {
        "source": source,
        "predicted_class": pred_class,
        "probabilities": probs_dict,
        "deficits": deficits,
        "suggestions": suggestions,
        "ph_advice": phinfo
    }
    return jsonify(resp)





# ---------------- Run ----------------
if __name__ == '__main__':
    print("Starting Flask app. Device for torch:", _DEVICE)
    app.run(debug=True)


