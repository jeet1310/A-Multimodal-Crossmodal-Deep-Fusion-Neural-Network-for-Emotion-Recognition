import cv2
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import joblib
import vosk, json, csv, time
from deepface import DeepFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path
from datetime import datetime
import sys
import pyttsx3   # for speech

# ================== CONFIG ==================
MODEL_PATH = r"C:\Users\jainj\OneDrive\Desktop\emotion_model_output1\best_model.h5"
SCALER_IMG_PATH = r"C:\Users\jainj\OneDrive\Desktop\capstone\emotion_model_output\scaler_img.joblib"
SCALER_AUD_PATH = r"C:\Users\jainj\OneDrive\Desktop\capstone\emotion_model_output\scaler_aud.joblib"
SCALER_TXT_PATH = r"C:\Users\jainj\OneDrive\Desktop\capstone\emotion_model_output\scaler_txt.joblib"

VOSK_MODEL_PATH = r"C:\Users\jainj\Downloads\vosk-model-small-en-us-0.15\vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
TARGET_SR = 22050
LOG_FILE = r"C:\Users\jainj\OneDrive\Desktop\capstone\realtime_log.csv"

N_MFCC = 40
N_CHROMA = 12
N_SPEC_CONTRAST = 7

# Emotion labels (match your training)
idx_to_label = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "sad", 5: "surprise", 6: "neutral"
}

# Emotional coach responses
coach_responses = {
    "angry": "I sense anger. Let's take a deep breath together.",
    "disgust": "You're feeling disgusted. It's okay to step away from what's bothering you.",
    "fear": "I sense fear. Remember, you're safe right now.",
    "happy": "I see happiness! Keep smiling, it looks good on you.",
    "sad": "I sense sadness. It's okay to feel down sometimes.",
    "surprise": "That was surprising! Unexpected things can be exciting too.",
    "neutral": "You seem neutral. Balanced and calm is always nice."
}

# ================== LOAD MODELS ==================
print("Loading model & scalers...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler_img = joblib.load(SCALER_IMG_PATH)
scaler_aud = joblib.load(SCALER_AUD_PATH)
scaler_txt = joblib.load(SCALER_TXT_PATH)
print("Model & scalers loaded.")

print("Loading Vosk...")
vosk_model = vosk.Model(VOSK_MODEL_PATH)
print("Vosk loaded.")

print("Loading sentiment pipeline...")
sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_pipe = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer, return_all_scores=True)
print("Sentiment pipeline loaded.")

# ================== AUDIO FEATURES ==================
def extract_audio_features(y, sr):
    if y is None or len(y) < 32:
        return np.zeros((1, scaler_aud.n_features_in_))

    y = np.asarray(y, dtype=np.float32)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    y = np.nan_to_num(y)

    pitches, mags = librosa.piptrack(y=y, sr=sr)
    valid = []
    if mags.size:
        for col in range(mags.shape[1]):
            mag_col = mags[:, col]
            if mag_col.any():
                idx = int(mag_col.argmax())
                p = pitches[idx, col]
                if not np.isnan(p) and p > 0:
                    valid.append(p)
    avg_pitch = float(np.mean(valid)) if valid else 0.0
    avg_energy = float(np.sum(np.abs(y) ** 2))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    avg_mfcc = np.mean(mfcc, axis=1) if mfcc.shape[1] > 0 else np.zeros(N_MFCC)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    avg_chroma = np.mean(chroma, axis=1) if chroma.shape[1] > 0 else np.zeros(N_CHROMA)

    spec = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=N_SPEC_CONTRAST-1)
    avg_spec = np.mean(spec, axis=1) if spec.shape[1] > 0 else np.zeros(N_SPEC_CONTRAST)

    features = np.concatenate(([avg_pitch, avg_energy], avg_mfcc, avg_chroma, avg_spec))
    n_expected = scaler_aud.n_features_in_
    if len(features) < n_expected:
        features = np.pad(features, (0, n_expected - len(features)))
    elif len(features) > n_expected:
        features = features[:n_expected]

    return scaler_aud.transform([features])

# ================== TEXT FEATURES ==================
def extract_text_features_from_transcript(transcript):
    if not transcript or not transcript.strip():
        return np.zeros((1, scaler_txt.n_features_in_))

    try:
        scores = sentiment_pipe(transcript[:512])[0]
        probs = [s["score"] for s in scores]
        probs = np.array(probs, dtype=float)
    except:
        probs = np.array([0.5, 0.5], dtype=float)

    if probs.size < scaler_txt.n_features_in_:
        probs = np.pad(probs, (0, scaler_txt.n_features_in_ - probs.size))
    elif probs.size > scaler_txt.n_features_in_:
        probs = probs[:scaler_txt.n_features_in_]

    return scaler_txt.transform([probs])

# ================== RECORD / STOP ==================
print("Controls: press 'r' to RECORD, 's' to STOP & PREDICT, 'q' to QUIT.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    sys.exit(1)

log_path = Path(LOG_FILE)
log_path.parent.mkdir(parents=True, exist_ok=True)
if not log_path.exists():
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "transcript", "predicted_emotion"])

recording = False
audio_collected = []
frames_collected = []

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status, flush=True)
    audio_collected.append(indata.copy().flatten())

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if recording:
            frames_collected.append(frame.copy())
            cv2.putText(frame, "REC", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.putText(frame, "Press 'r'=Record / 's'=Stop & Predict / 'q'=Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Realtime Multimodal (Record/Stop)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording:
            audio_collected.clear()
            frames_collected.clear()
            try:
                stream.start()
                recording = True
                start_time = time.time()
                print("Recording started...")
            except Exception as e:
                print("Failed to start audio stream:", e)

        elif key == ord('s') and recording:
            recording = False
            stream.stop()
            duration = time.time() - start_time
            print(f"Recording stopped. Duration {duration:.2f}s. Processing...")

            if len(audio_collected) == 0:
                print("No audio captured.")
                continue
            audio_np = np.concatenate(audio_collected, axis=0).astype(np.float32)

            # AUDIO features
            aud_feat = extract_audio_features(audio_np, SAMPLE_RATE)

            # IMAGE features via DeepFace
            if frames_collected:
                emo_accum = []
                for f in frames_collected:
                    try:
                        analysis = DeepFace.analyze(f, actions=['emotion'], enforce_detection=False)
                        emotions = analysis[0]["emotion"]
                        emo_accum.append(list(emotions.values()))
                    except Exception as e:
                        print("DeepFace error:", e)
                if emo_accum:
                    img_vec = np.mean(np.array(emo_accum), axis=0)
                    n_img_expected = scaler_img.n_features_in_
                    if img_vec.size < n_img_expected:
                        img_vec = np.pad(img_vec, (0, n_img_expected - img_vec.size))
                    elif img_vec.size > n_img_expected:
                        img_vec = img_vec[:n_img_expected]
                    img_feat = scaler_img.transform([img_vec])
                else:
                    img_feat = np.zeros((1, scaler_img.n_features_in_))
            else:
                img_feat = np.zeros((1, scaler_img.n_features_in_))

            # TRANSCRIPT
            try:
                audio_clipped = np.clip(audio_np, -1.0, 1.0)
                audio_int16 = (audio_clipped * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                rec_local = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
                rec_local.AcceptWaveform(audio_bytes)
                result_json = json.loads(rec_local.FinalResult())
                transcript = result_json.get("text", "").strip()
            except Exception as e:
                print("Vosk error:", e)
                transcript = ""

            txt_feat = extract_text_features_from_transcript(transcript)

            # PREDICT
            try:
                pred = model.predict([img_feat, aud_feat, txt_feat], verbose=0)
                pred_class = int(np.argmax(pred, axis=1)[0])
                emotion = idx_to_label.get(pred_class, "unknown")
            except Exception as e:
                print("Prediction error:", e)
                emotion = "error"

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([ts, transcript, emotion])
            print(f"[{ts}] Transcript: '{transcript}' --> Predicted: {emotion}")

            # COACH RESPONSE
            coach_line = coach_responses.get(emotion, "I'm here with you.")
            print("Coach:", coach_line)

            try:
                tts_engine = pyttsx3.init()
                tts_engine.say(coach_line)
                tts_engine.runAndWait()
                del tts_engine
            except Exception as e:
                print("Speech error:", e)

            disp = frames_collected[-1] if frames_collected else frame.copy()
            cv2.putText(disp, f"Emotion: {emotion}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            if transcript:
                cv2.putText(disp, transcript[:80], (30, disp.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(disp, coach_line[:80], (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

            cv2.imshow("Realtime Multimodal (Record/Stop)", disp)
            cv2.waitKey(1000)

            audio_collected.clear()
            frames_collected.clear()

        elif key == ord('q'):
            print("Quitting...")
            break

except KeyboardInterrupt:
    print("KeyboardInterrupt received. Exiting...")

finally:
    try:
        if stream.active:
            stream.stop()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
    print("Done. Log saved to:", LOG_FILE)
