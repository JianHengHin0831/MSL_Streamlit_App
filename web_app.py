import base64
import io
import logging
import os
import threading
import time
from collections import Counter
from types import SimpleNamespace

import cv2
import flask
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from PIL import Image


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

process_lock = threading.Lock()
_last_ts_lock = threading.Lock()
_last_timestamp_ms = 0

app = flask.Flask(__name__)

GESTURES = np.sort(np.array([
    'abang', 'ada', 'ambil', 'anak_lelaki', 'anak_perempuan', 'apa', 'apa_khabar', 'arah',
    'assalamualaikum', 'ayah', 'bagaimana', 'bahasa_isyarat', 'baik', 'baik_2', 'baca',
    'bapa', 'bapa_saudara', 'bas', 'bawa', 'beli', 'beli_2', 'berapa', 'berjalan', 'berlari',
    'bila', 'bola', 'boleh', 'bomba', 'buang', 'buat', 'curi', 'dapat', 'dari', 'emak',
    'emak_saudara', 'hari', 'hi', 'hujan', 'jahat', 'jam', 'jangan', 'jumpa', 'kacau',
    'kakak', 'keluarga', 'kereta', 'kesakitan', 'lelaki', 'lemak', 'lupa', 'main', 'makan',
    'mana', 'marah', 'mari', 'masa', 'masalah', 'minum', 'mohon', 'nasi', 'nasi_lemak',
    'panas', 'panas_2', 'pandai', 'pandai_2', 'payung', 'pen', 'pensil', 'perempuan',
    'pergi', 'pergi_2', 'perlahan', 'perlahan_2', 'pinjam', 'polis', 'pukul', 'ribut',
    'sampai', 'saudara', 'sejuk', 'sekolah', 'siapa', 'sudah', 'suka', 'tandas', 'tanya',
    'teh_tarik', 'teksi', 'tidur', 'tolong'
]))

CONFIG = SimpleNamespace(
    quantized_model='msl_model_quant.onnx',
    fallback_model='msl_model.onnx',
    min_sequence_length=20,
    prediction_window=6,
    prediction_threshold=3,
    min_confidence=0.45,
    use_hands_only=True
)


def load_onnx_session():
    model_path = CONFIG.quantized_model if os.path.exists(CONFIG.quantized_model) else CONFIG.fallback_model
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
        logger.info("Loaded ONNX model: %s", model_path)
        return session
    except Exception as exc:
        logger.exception("Failed to load ONNX model: %s", exc)
        return None


def init_mediapipe():
    drawing = mp.solutions.drawing_utils
    if CONFIG.use_hands_only:
        try:
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe initialized in hands-only mode")
            return SimpleNamespace(mode='hands', hands=hands, holistic=None, drawing=drawing)
        except Exception as exc:
            logger.warning("Hands-only mode failed, falling back to holistic: %s", exc)

    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    logger.info("MediaPipe holistic mode active")
    return SimpleNamespace(mode='holistic', hands=None, holistic=holistic, drawing=drawing)


ONNX_SESSION = load_onnx_session()
MEDIA_PIPE = init_mediapipe()


def extract_keypoints(results):
    pose = np.zeros(25 * 4)
    if getattr(results, 'pose_landmarks', None):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark[:25]]).flatten()

    left = np.zeros(21 * 3)
    if getattr(results, 'left_hand_landmarks', None):
        left = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()

    right = np.zeros(21 * 3)
    if getattr(results, 'right_hand_landmarks', None):
        right = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, left, right])


def pad_sequence(sequence):
    window = np.array(sequence[-30:])
    if window.shape[0] < 30:
        pad = np.zeros((30 - window.shape[0], window.shape[1]), dtype=window.dtype)
        window = np.concatenate([pad, window], axis=0)
    return np.expand_dims(window, axis=0).astype(np.float32)


def predict_gesture(sequence):
    if ONNX_SESSION is None or len(sequence) < CONFIG.min_sequence_length:
        return None, 0.0

    input_data = pad_sequence(sequence)
    input_name = ONNX_SESSION.get_inputs()[0].name
    output_name = ONNX_SESSION.get_outputs()[0].name
    output = ONNX_SESSION.run([output_name], {input_name: input_data})[0][0]
    probs = np.exp(output - np.max(output))
    probs /= np.sum(probs)
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])


def empty_state():
    return {
        'sequence': [],
        'predictions': [],
        'sentence': 'Waiting...',
        'frame_count': 0
    }


global_state = empty_state()


def ensure_monotonic_timestamp():
    global _last_timestamp_ms
    with _last_ts_lock:
        ts = int(time.time() * 1000)
        if ts <= _last_timestamp_ms:
            ts = _last_timestamp_ms + 1
        _last_timestamp_ms = ts
    return ts


def process_with_hands(image_rgb):
    mp_results = MEDIA_PIPE.hands.process(image_rgb)
    left = right = None
    if getattr(mp_results, 'multi_hand_landmarks', None):
        for i, lm in enumerate(mp_results.multi_hand_landmarks):
            label = None
            try:
                label = mp_results.multi_handedness[i].classification[0].label.lower()
            except Exception:
                pass
            if label == 'left' and left is None:
                left = lm
            elif label == 'right' and right is None:
                right = lm
            elif left is None:
                left = lm
            else:
                right = lm

    return SimpleNamespace(pose_landmarks=None, left_hand_landmarks=left, right_hand_landmarks=right)


def process_with_holistic(image_rgb, timestamp_ms):
    try:
        return MEDIA_PIPE.holistic.process(image_rgb, timestamp_ms=timestamp_ms)
    except TypeError:
        return MEDIA_PIPE.holistic.process(image_rgb)


def decode_image_from_request():
    content_type = flask.request.content_type or ''
    if content_type.startswith('image/'):
        return flask.request.get_data()
    payload = flask.request.get_json(silent=True) or {}
    image_data = payload.get('image')
    if not image_data:
        return None
    image_data = image_data.split(',')[1] if ',' in image_data else image_data
    return base64.b64decode(image_data)


def update_sentence(state, pred_idx, confidence):
    state['predictions'].append(pred_idx)
    state['predictions'] = state['predictions'][-CONFIG.prediction_window:]

    if not state['predictions']:
        return

    most_common_id, frequency = Counter(state['predictions']).most_common(1)[0]
    if frequency >= CONFIG.prediction_threshold and confidence > CONFIG.min_confidence:
        state['sentence'] = GESTURES[most_common_id]


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        global global_state
        start_time = time.time()

        image_bytes = decode_image_from_request()
        if not image_bytes:
            return flask.jsonify({'error': 'No image data provided'}), 400

        frame = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        frame = np.array(frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        with process_lock:
            timestamp_ms = ensure_monotonic_timestamp()
            if MEDIA_PIPE.mode == 'hands':
                mp_results = process_with_hands(frame_rgb)
            else:
                mp_results = process_with_holistic(frame_rgb, timestamp_ms)
        frame_rgb.flags.writeable = True

        if mp_results.left_hand_landmarks:
            MEDIA_PIPE.drawing.draw_landmarks(frame_bgr, mp_results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        if mp_results.right_hand_landmarks:
            MEDIA_PIPE.drawing.draw_landmarks(frame_bgr, mp_results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        global_state['frame_count'] += 1
        keypoints = extract_keypoints(mp_results)
        global_state['sequence'].append(keypoints)
        global_state['sequence'] = global_state['sequence'][-30:]

        if len(global_state['sequence']) >= CONFIG.min_sequence_length:
            pred_idx, confidence = predict_gesture(global_state['sequence'])
            if pred_idx is not None:
                update_sentence(global_state, pred_idx, confidence)
                if confidence > CONFIG.min_confidence:
                    cv2.rectangle(frame_bgr, (0, 0), (int(confidence * 200), 40), (0, 255, 0), -1)

        cv2.rectangle(frame_bgr, (0, 40), (640, 80), (245, 117, 16), -1)
        cv2.putText(frame_bgr, global_state['sentence'], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame_bgr)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        proc_time_ms = int((time.time() - start_time) * 1000)

        return flask.jsonify({'image': 'data:image/jpeg;base64,' + image_base64,
                              'prediction': global_state['sentence'],
                              'proc_ms': proc_time_ms})

    except Exception as exc:
        logger.exception("process_frame failed: %s", exc)
        return flask.jsonify({'error': str(exc)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    global global_state
    global_state = empty_state()
    return flask.jsonify({'status': 'reset'})


if __name__ == '__main__':
    logger.info("Starting Flask app on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)