// YOLOv8 via ONNX Runtime Web — multiple CDN fallbacks
const ONNX_RUNTIME_URLS = [
  "https://unpkg.com/onnxruntime-web@1.18.0/dist/ort.web.min.js",
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.web.min.js",
  "https://esm.sh/onnxruntime-web@1.18.0/dist/ort.web.min.js",
];
const YOLOV8_MODEL_URLS = [
  "https://huggingface.co/Xenova/yolov8m-coco/resolve/main/model.onnx",
  "https://huggingface.co/Xenova/yolov8n-coco/resolve/main/model.onnx",
];

// ============================================================================
// CONFIG
// ============================================================================

const CONFIG = {
  // Detection
  minConfidence: 0.5,
  detectionInterval: 100, // ms between detection frames
  detectModel: "lite2", // "lite2" or "lite0"; fallback to lite0 if lite2 exceeds latency budget

  // Camera
  preferredFacingMode: "environment", // rear camera on phones; falls back to any camera on desktop

  // Distance estimation
  focalLengthPx: 400, // approximate focal length for a typical phone camera
  objectSizes: {
    // Average real-world sizes per COCO class (in meters)
    "person": 1.7,
    "bicycle": 1.5,
    "car": 1.8,
    "motorcycle": 1.8,
    "airplane": 35,
    "bus": 3.0,
    "train": 25,
    "truck": 2.5,
    "boat": 8,
    "traffic light": 0.5,
    "fire hydrant": 0.6,
    "stop sign": 0.7,
    "parking meter": 2.0,
    "bench": 1.5,
    "cat": 0.25,
    "dog": 0.5,
    "horse": 2.0,
    "sheep": 1.5,
    "cow": 1.8,
    "elephant": 3.0,
    "bear": 2.0,
    "zebra": 2.5,
    "giraffe": 5.0,
    "backpack": 0.35,
    "umbrella": 0.7,
    "handbag": 0.3,
    "tie": 1.4,
    "suitcase": 0.6,
    "frisbee": 0.27,
    "skis": 1.65,
    "snowboard": 1.5,
    "sports ball": 0.24,
    "kite": 0.7,
    "baseball bat": 0.85,
    "baseball glove": 0.25,
    "skateboard": 0.8,
    "surfboard": 1.8,
    "tennis racket": 0.68,
    "bottle": 0.25,
    "wine glass": 0.15,
    "cup": 0.08,
    "fork": 0.20,
    "knife": 0.25,
    "spoon": 0.20,
    "bowl": 0.25,
    "banana": 0.19,
    "apple": 0.08,
    "sandwich": 0.15,
    "orange": 0.08,
    "broccoli": 0.15,
    "carrot": 0.20,
    "hot dog": 0.15,
    "pizza": 0.30,
    "donut": 0.08,
    "cake": 0.20,
    "chair": 0.9,
    "couch": 2.0,
    "potted plant": 0.5,
    "bed": 1.6,
    "dining table": 1.5,
    "toilet": 0.4,
    "tv": 0.6,
    "laptop": 0.35,
    "mouse": 0.08,
    "remote": 0.15,
    "keyboard": 0.45,
    "microwave": 0.5,
    "oven": 0.7,
    "toaster": 0.25,
    "sink": 0.8,
    "refrigerator": 1.7,
    "book": 0.20,
    "clock": 0.20,
    "vase": 0.25,
    "scissors": 0.20,
    "teddy bear": 0.4,
    "hair drier": 0.25,
    "toothbrush": 0.20,
    "default": 0.5, // fallback for unmapped classes
  },

  // Urgency tiers (distance in meters)
  urgencyTiers: {
    critical: 1.2, // < 1.2m
    near: 3.0, // 1.2m - 3m
    far: Infinity, // > 3m
  },

  // Announcement logic
  maxObjectsPerCycle: 3,
  heartbeatInterval: 5000, // repeat closest object every 5s if nothing changed (ms)
  veryCloseThreshold: 0.7, // interrupt speech if object closer than this
  distanceBucketSize: 0.5, // only re-announce if distance changes by at least this much
  // Detection runs ~10x/second. Without these two floors the same object is
  // re-spoken every frame, which cancels the previous utterance mid-word and
  // comes out as a stutter.
  minRepeatInterval: 1400, // never repeat the same label faster than this (ms)
  interruptCooldown: 2000, // and don't re-fire the very-close warning faster than this

  // Audio
  panningToneDuration: 150, // ms
  panningToneFrequency: 800, // Hz
  panningToneVolume: 0.3,
};

// ============================================================================
// STATE
// ============================================================================

let state = {
  scanning: false,
  cameraActive: false,
  modelLoaded: false,
  videoElement: null,
  onnxSession: null, // YOLOv8 ONNX Runtime session
  audioContext: null,
  lastAnnouncements: new Map(),
  lastInterrupt: { label: null, time: 0 },
  detectionStartTime: 0,
  lastDetections: [],
  cameras: [],
  activeDeviceId: null,
  cameraWanted: false,
};

// ============================================================================
// UI ELEMENTS
// ============================================================================

const ui = {
  toggleButton: document.getElementById("toggle-button"),
  actionText: document.getElementById("action-text"),
  hintText: document.getElementById("hint-text"),
  statusMessage: document.getElementById("status-message"),
  videoElement: document.getElementById("camera-feed"),
  canvas: document.getElementById("detection-canvas"),
  cameraInfo: document.getElementById("camera-info"),
  switchCamera: document.getElementById("switch-camera"),
};

// The button carries the whole interface, so its three states are defined in
// one place: what it says, what it hints, and how a screen reader announces it.
function setButtonState(mode) {
  const modes = {
    start: {
      action: "Tap to start",
      hint: "Anywhere on the screen",
      label: "Start scanning for objects",
    },
    stop: {
      action: "Tap to stop",
      hint: "Tap anywhere to stop",
      label: "Stop scanning",
    },
    unavailable: {
      action: "Camera needed",
      hint: "Allow camera access, then reload",
      label: "Camera unavailable. Allow camera access and reload the page.",
    },
  };

  const next = modes[mode];
  ui.actionText.textContent = next.action;
  ui.hintText.textContent = next.hint;
  ui.toggleButton.setAttribute("aria-label", next.label);
  document.body.classList.toggle("scanning", mode === "stop");
  ui.toggleButton.disabled = mode === "unavailable";
}

// ============================================================================
// INIT
// ============================================================================

async function initApp() {
  ui.toggleButton.addEventListener("click", toggleScanning);
  ui.switchCamera.addEventListener("click", switchCamera);
  state.videoElement = ui.videoElement;

  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraInfo("getUserMedia unavailable — needs HTTPS or localhost");
    updateStatus("Camera unavailable on this page", { error: true });
    setButtonState("unavailable");
    return;
  }

  // Automatically request camera access on page load
  setButtonState("start");
  updateStatus("Asking for camera permission");
  try {
    await requestCamera();
    updateStatus("Ready");
    speak("Ready. Tap anywhere to start.");
  } catch (err) {
    console.error("Camera request failed:", err);
    setButtonState("unavailable");
  }
}

function initAudioContext() {
  // Initialize Web Audio API only after user gesture (autoplay policy)
  if (!state.audioContext) {
    try {
      state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) {
      console.warn("Web Audio API not available:", e);
      return;
    }
  }
  if (state.audioContext.state === "suspended") {
    state.audioContext.resume();
  }
}

// ============================================================================
// CAMERA ACCESS
// ============================================================================

async function refreshCameraList() {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  // Labels are only populated once permission has been granted
  state.cameras = devices.filter((d) => d.kind === "videoinput");
  const multiple = state.cameras.length > 1;
  ui.switchCamera.hidden = !multiple;
  document.body.classList.toggle("has-switch", multiple);
  return state.cameras;
}

// Try progressively looser constraints so the same code works on a phone
// (rear camera) and on a laptop (built-in front camera, no facingMode support).
function buildConstraintAttempts(deviceId) {
  const size = { width: { ideal: 640 }, height: { ideal: 480 } };
  const attempts = [];

  if (deviceId) {
    attempts.push({ video: { deviceId: { exact: deviceId }, ...size } });
  }
  attempts.push({ video: { facingMode: { ideal: CONFIG.preferredFacingMode }, ...size } });
  attempts.push({ video: { facingMode: { ideal: "user" }, ...size } });
  attempts.push({ video: size });
  attempts.push({ video: true });

  return attempts;
}

async function requestCamera(deviceId = null) {
  updateStatus("Asking for camera permission");
  setCameraInfo("requesting camera…");

  // Enumerate before permission so we can report "no camera attached" distinctly
  // from "permission denied" — labels stay blank until access is granted.
  await refreshCameraList().catch(() => {});
  if (state.cameras.length === 0 && navigator.mediaDevices?.enumerateDevices) {
    console.warn("No videoinput devices reported before permission prompt");
  }

  let stream = null;
  let lastErr = null;

  for (const constraints of buildConstraintAttempts(deviceId)) {
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      break;
    } catch (err) {
      lastErr = err;
      // A hard denial will not be fixed by loosening constraints — stop early.
      if (err.name === "NotAllowedError" || err.name === "SecurityError") break;
      console.warn("Camera constraint attempt failed:", constraints, err.name);
    }
  }

  if (!stream) {
    const errorMsg = describeCameraError(lastErr);
    setCameraInfo(lastErr?.name || "camera error");
    updateStatus(errorMsg, { error: true });
    speak(errorMsg);
    throw lastErr || new Error("Camera unavailable");
  }

  state.videoElement.srcObject = stream;

  const track = stream.getVideoTracks()[0];
  const settings = track?.getSettings?.() || {};
  state.activeDeviceId = settings.deviceId || null;

  // Re-enumerate now that permission is granted, so device labels are readable
  await refreshCameraList().catch(() => {});

  await waitForVideoReady(state.videoElement);

  state.cameraActive = true;
  state.cameraWanted = true;

  const label = track?.label || "Camera";
  const facing = settings.facingMode ? ` · ${settings.facingMode}` : "";
  const res = `${state.videoElement.videoWidth}×${state.videoElement.videoHeight}`;
  const fps = settings.frameRate ? ` @ ${Math.round(settings.frameRate)}fps` : "";
  setCameraInfo(`${label} · ${res}${fps}${facing}`);

  console.log("Camera active:", { label, settings, resolution: res });

  updateStatus("Camera ready");
}

function waitForVideoReady(video) {
  if (video.readyState >= video.HAVE_CURRENT_DATA && video.videoWidth > 0) {
    return video.play().catch(() => {});
  }
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Timed out waiting for video metadata")), 10000);
    video.onloadedmetadata = () => {
      clearTimeout(timer);
      video.play().catch(() => {});
      resolve();
    };
  });
}

function describeCameraError(err) {
  switch (err?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission denied. Please allow camera access.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera found on this device.";
    case "NotReadableError":
      return "Camera is in use by another app. Close it and reload.";
    default:
      return "Camera not available. Check your device.";
  }
}

async function switchCamera(event) {
  // The tap area sits underneath; don't toggle scanning as well.
  event?.stopPropagation();
  if (state.cameras.length < 2) return;

  const idx = state.cameras.findIndex((c) => c.deviceId === state.activeDeviceId);
  const next = state.cameras[(idx + 1) % state.cameras.length];
  stopCamera();
  try {
    await requestCamera(next.deviceId);
    speak("Camera switched");
  } catch (err) {
    console.error("Camera switch failed:", err);
  }
}

function stopCamera() {
  if (state.videoElement.srcObject) {
    state.videoElement.srcObject.getTracks().forEach((track) => track.stop());
    state.videoElement.srcObject = null;
  }
  state.cameraActive = false;
  clearOverlay();
  setCameraInfo("camera off");
}

// Sighted-testing diagnostics only — this line is aria-hidden in the markup.
function setCameraInfo(text) {
  ui.cameraInfo.textContent = text;
}

// ============================================================================
// MODEL LOADING
// ============================================================================

async function loadModel() {
  if (state.modelLoaded) return;

  try {
    updateStatus("Getting ready");

    // Try multiple CDNs for ONNX Runtime
    let ort = null;
    let lastRuntimeErr = null;

    for (const runtimeUrl of ONNX_RUNTIME_URLS) {
      try {
        console.log(`Trying ONNX Runtime from: ${runtimeUrl}`);
        ort = await import(runtimeUrl);
        console.log("ONNX Runtime loaded");
        break;
      } catch (err) {
        console.warn(`ONNX Runtime failed (${runtimeUrl}):`, err.message);
        lastRuntimeErr = err;
      }
    }

    if (!ort) {
      throw lastRuntimeErr || new Error("Could not load ONNX Runtime from any CDN");
    }

    // Initialize ONNX Runtime
    if (ort.env?.wasm) {
      ort.env.wasm.simdSupported = true;
      ort.env.wasm.numThreads = 1;
    }

    // Try each model URL until one works
    let lastModelErr = null;
    for (const modelUrl of YOLOV8_MODEL_URLS) {
      try {
        console.log(`Trying model: ${modelUrl}`);
        updateStatus("Downloading AI model...");

        state.onnxSession = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });

        console.log("Model loaded successfully");
        state.modelLoaded = true;
        return;
      } catch (err) {
        console.warn(`Model failed (${modelUrl}):`, err.message);
        lastModelErr = err;
      }
    }

    throw lastModelErr || new Error("No model URLs available");
  } catch (err) {
    console.error("Model load failed:", err);
    const errorMsg = `Could not load AI model. Check your internet. Error: ${err.message}`;
    updateStatus(errorMsg, { error: true });
    speak(errorMsg);
    throw err;
  }
}

// ============================================================================
// DETECTION & ANNOUNCEMENT LOGIC
// ============================================================================

async function startScanning() {
  // Initialize audio context after user gesture
  initAudioContext();

  // Flip the screen before any awaiting. The first run has to fetch the wasm
  // runtime and the model, which takes seconds — waiting for that before
  // clearing the prompt made the tap feel unresponsive.
  state.scanning = true;
  state.detectionStartTime = Date.now();
  setButtonState("stop");
  speak("Starting. Tap anywhere to stop.");

  try {
    if (!state.cameraActive) {
      await requestCamera(state.activeDeviceId);
    }
    if (!state.scanning) return; // stopped while the camera was opening

    await loadModel();
    if (!state.scanning) return; // stopped while the model was downloading

    updateStatus("Scanning");
    detectionLoop();
  } catch (err) {
    console.error("Failed to start scanning:", err);
    state.scanning = false;
    setButtonState("start");
  }
}

function stopScanning() {
  state.scanning = false;
  state.lastAnnouncements.clear();
  state.lastInterrupt = { label: null, time: 0 };
  clearOverlay();
  speechSynthesis.cancel();
  setButtonState("start");
  updateStatus("Stopped");
  // Always name the next action: without this a blind user is left on a
  // silent screen with no clue that tapping again restarts it.
  speak("Stopped. Tap anywhere to start.");

  // The camera deliberately stays open, matching the state the page loads in,
  // so tapping start again shows the picture immediately instead of paying
  // for the permission and warm-up round trip a second time.
}

function toggleScanning() {
  if (state.scanning) {
    stopScanning();
  } else {
    startScanning();
  }
}

// COCO class names for YOLOv8
const COCO_CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "cat", "dog",
  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
  "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
  "keyboard", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
  "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

async function detectWithYOLOv8(videoElement) {
  if (!state.onnxSession) throw new Error("Model not loaded");

  const ort = await import(`${ONNX_RUNTIME_CDN}/ort.web.min.js`);

  // Prepare input: letterbox resize to 640x640
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 640;
  const ctx = canvas.getContext("2d");

  // Letterbox: scale and pad to 640x640
  const scale = Math.min(640 / videoElement.videoWidth, 640 / videoElement.videoHeight);
  const x = (640 - videoElement.videoWidth * scale) / 2;
  const y = (640 - videoElement.videoHeight * scale) / 2;

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, 640, 640);
  ctx.drawImage(
    videoElement,
    x,
    y,
    videoElement.videoWidth * scale,
    videoElement.videoHeight * scale
  );

  // Extract image data and normalize
  const imageData = ctx.getImageData(0, 0, 640, 640);
  const data = imageData.data;
  const input = new Float32Array(3 * 640 * 640);

  for (let i = 0; i < data.length; i += 4) {
    input[i / 4] = data[i] / 255;                    // R
    input[i / 4 + 640 * 640] = data[i + 1] / 255;   // G
    input[i / 4 + 2 * 640 * 640] = data[i + 2] / 255; // B
  }

  // Run inference
  const inputTensor = new ort.Tensor("float32", input, [1, 3, 640, 640]);
  const results = await state.onnxSession.run({ images: inputTensor });

  // Parse YOLOv8 output (1, 84, 8400)
  const output = results.output0.data;
  const detections = [];

  for (let i = 0; i < 8400; i++) {
    const conf = output[4 * 8400 + i]; // Objectness score
    if (conf < CONFIG.minConfidence) continue;

    const x = output[0 * 8400 + i];
    const y = output[1 * 8400 + i];
    const w = output[2 * 8400 + i];
    const h = output[3 * 8400 + i];

    let maxClass = 0,
      maxScore = 0;
    for (let c = 0; c < 80; c++) {
      const score = output[(5 + c) * 8400 + i];
      if (score > maxScore) {
        maxScore = score;
        maxClass = c;
      }
    }

    const finalScore = conf * maxScore;
    if (finalScore < CONFIG.minConfidence) continue;

    // Convert from letterbox coords to original video coords
    const origX = ((x - x) / scale) - x;
    const origY = ((y - y) / scale) - y;
    const origW = (w / scale);
    const origH = (h / scale);

    detections.push({
      boundingBox: {
        originX: (origX - origW / 2) / videoElement.videoWidth,
        originY: (origY - origH / 2) / videoElement.videoHeight,
        width: origW / videoElement.videoWidth,
        height: origH / videoElement.videoHeight,
      },
      categories: [{ categoryName: COCO_CLASSES[maxClass], score: finalScore }],
    });
  }

  return detections;
}

async function detectionLoop() {
  if (!state.scanning) return;

  const now = Date.now();

  // Detect objects
  if (
    state.videoElement.readyState === state.videoElement.HAVE_ENOUGH_DATA &&
    state.modelLoaded
  ) {
    try {
      const detections = await detectWithYOLOv8(state.videoElement);

      if (detections && detections.length > 0) {
        state.lastDetections = detections;
        drawOverlay(detections);
        processDetections(detections);
      } else {
        state.lastDetections = [];
        clearOverlay();
      }
    } catch (err) {
      console.error("Detection error:", err);
    }
  }

  // Schedule next detection
  setTimeout(() => detectionLoop(), CONFIG.detectionInterval);
}

// MediaPipe Tasks Vision reports boundingBox in PIXELS of the input frame.
// Guard against builds that hand back normalized values instead.
function toPixelBox(bbox, videoWidth, videoHeight) {
  const looksNormalized =
    bbox.width <= 1 && bbox.height <= 1 && bbox.originX <= 1 && bbox.originY <= 1;

  return looksNormalized
    ? {
        x: bbox.originX * videoWidth,
        y: bbox.originY * videoHeight,
        w: bbox.width * videoWidth,
        h: bbox.height * videoHeight,
      }
    : { x: bbox.originX, y: bbox.originY, w: bbox.width, h: bbox.height };
}

function processDetections(detections) {
  const now = Date.now();
  const videoWidth = state.videoElement.videoWidth || 1;
  const videoHeight = state.videoElement.videoHeight || 1;

  // Compute distances and urgency tiers
  const objectsWithData = detections
    .map((d) => {
      const label = d.categories[0]?.categoryName || "unknown";
      const confidence = d.categories[0]?.score || 0;

      const box = toPixelBox(d.boundingBox, videoWidth, videoHeight);
      const bboxAvgSize = (box.w + box.h) / 2;

      const realSize = CONFIG.objectSizes[label] || CONFIG.objectSizes.default;
      const distance = (realSize * CONFIG.focalLengthPx) / bboxAvgSize;

      const centerXNorm = (box.x + box.w / 2) / videoWidth;
      const direction = centerXNorm < 0.33 ? "left" : centerXNorm > 0.67 ? "right" : "center";

      let tier = "far";
      if (distance < CONFIG.urgencyTiers.critical) tier = "critical";
      else if (distance < CONFIG.urgencyTiers.near) tier = "near";

      return { label, distance, confidence, direction, tier, centerXNorm, box };
    })
    .sort((a, b) => a.distance - b.distance); // Sort by distance

  // Check for very close object (interrupt). Barging in on every frame would
  // cancel the previous utterance before it finished a word, so this fires
  // only for a newly-close object or after the cooldown.
  const veryClose = objectsWithData.find((o) => o.distance < CONFIG.veryCloseThreshold);
  if (veryClose) {
    const isNewThreat = veryClose.label !== state.lastInterrupt.label;
    const cooledDown = now - state.lastInterrupt.time >= CONFIG.interruptCooldown;

    if (isNewThreat || cooledDown) {
      state.lastInterrupt = { label: veryClose.label, time: now };
      speechSynthesis.cancel();
      const warning = `${veryClose.label} very close`;
      updateStatus(warning);
      speak(warning, { rate: 1.2, pitch: 1.1 });
      playPanningTone(veryClose.centerXNorm);
      state.lastAnnouncements.set(veryClose.label, { distance: veryClose.distance, time: now });
    }
    return; // Nothing outranks an imminent collision
  }

  // Build urgency queue
  const queue = [];
  const tierOrder = ["critical", "near", "far"];

  for (const tier of tierOrder) {
    const tieredObjects = objectsWithData.filter((o) => o.tier === tier);
    for (const obj of tieredObjects) {
      if (queue.length >= CONFIG.maxObjectsPerCycle) break;
      queue.push(obj);
    }
    if (queue.length >= CONFIG.maxObjectsPerCycle) break;
  }

  // Determine what to announce
  const toAnnounce = queue.filter((obj) => {
    const lastRecord = state.lastAnnouncements.get(obj.label);

    if (lastRecord === undefined) return true; // New object

    // Hard floor first: distance readings jitter frame to frame, and without
    // this the bucket test below trips almost every cycle.
    if (now - lastRecord.time < CONFIG.minRepeatInterval) return false;

    if (Math.abs(obj.distance - lastRecord.distance) >= CONFIG.distanceBucketSize)
      return true; // Distance changed meaningfully
    if (now - lastRecord.time >= CONFIG.heartbeatInterval)
      return true; // Periodic repeat per object

    return false;
  });

  if (toAnnounce.length === 0) return;

  // The status line shows only the most urgent object; showing each in turn
  // made the text flicker several times a second.
  updateStatus(formatAnnouncement(toAnnounce[0]));

  toAnnounce.forEach((obj, i) => {
    // speechSynthesis queues utterances itself — no manual staggering needed.
    speak(formatAnnouncement(obj));
    // Tones would otherwise all land on the same instant and sum into a blip.
    playPanningTone(obj.centerXNorm, i * 0.18);
    state.lastAnnouncements.set(obj.label, { distance: obj.distance, time: now });
  });
}

function formatAnnouncement(obj) {
  const distStr = formatDistance(obj.distance);
  const dirStr = obj.direction === "center" ? "ahead" : `to your ${obj.direction}`;
  return `${obj.label}, ${distStr}, ${dirStr}`;
}

function formatDistance(meters) {
  if (meters < 1) return `${Math.round(meters * 100)} centimeters`;
  if (meters < 10) return `${meters.toFixed(1)} meters`;
  return `${Math.round(meters)} meters`;
}

// ============================================================================
// DEBUG OVERLAY
// ============================================================================

function drawOverlay(detections) {
  const video = state.videoElement;
  const canvas = ui.canvas;
  if (!video.videoWidth) return;

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = Math.max(2, canvas.width / 200);
  ctx.font = `${Math.max(14, canvas.width / 32)}px -apple-system, sans-serif`;
  ctx.textBaseline = "top";

  for (const d of detections) {
    const box = toPixelBox(d.boundingBox, canvas.width, canvas.height);
    const label = d.categories[0]?.categoryName || "unknown";
    const score = Math.round((d.categories[0]?.score || 0) * 100);
    const text = `${label} ${score}%`;

    ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    ctx.strokeRect(box.x, box.y, box.w, box.h);

    const metrics = ctx.measureText(text);
    const textHeight = parseInt(ctx.font, 10) * 1.3;
    ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
    ctx.fillRect(box.x, Math.max(0, box.y - textHeight), metrics.width + 10, textHeight);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, box.x + 5, Math.max(0, box.y - textHeight) + 2);
  }
}

function clearOverlay() {
  const ctx = ui.canvas.getContext("2d");
  ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
}

// ============================================================================
// AUDIO SYNTHESIS
// ============================================================================

function speak(text, options = {}) {
  if (!("speechSynthesis" in window)) {
    console.warn("Speech Synthesis not available");
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = options.rate || 1.0;
  utterance.pitch = options.pitch || 1.0;
  utterance.volume = 1.0;

  // Speech before the first interaction is refused on iOS and some Android
  // builds. Attempt it anyway — the greeting is worth having where it is
  // allowed — and swallow the refusal instead of logging an error for a
  // condition that is expected and unfixable. Screen reader users still get
  // the prompt from the button label and the live region.
  utterance.onerror = (event) => {
    if (event.error !== "not-allowed" && event.error !== "interrupted") {
      console.warn("Speech failed:", event.error);
    }
  };

  speechSynthesis.speak(utterance);
}

function playPanningTone(centerX, delaySeconds = 0) {
  if (!state.audioContext) return;

  const panValue = (centerX - 0.5) * 2; // -1 (left) to +1 (right)

  try {
    const now = state.audioContext.currentTime + delaySeconds;
    const duration = CONFIG.panningToneDuration / 1000;

    // Oscillator for tone
    const osc = state.audioContext.createOscillator();
    osc.frequency.value = CONFIG.panningToneFrequency;

    // Gain for volume
    const gain = state.audioContext.createGain();
    gain.gain.setValueAtTime(CONFIG.panningToneVolume, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    // Stereo panner
    const panner = state.audioContext.createStereoPanner();
    panner.pan.value = panValue;

    // Connect and play
    osc.connect(panner);
    panner.connect(gain);
    gain.connect(state.audioContext.destination);

    osc.start(now);
    osc.stop(now + duration);
  } catch (err) {
    console.warn("Failed to play panning tone:", err);
  }
}

// ============================================================================
// UI UPDATES
// ============================================================================

function updateStatus(text, { error = false } = {}) {
  // Rewriting an aria-live region with identical text makes some screen
  // readers announce it again, so bail out when nothing actually changed.
  if (ui.statusMessage.textContent === text) return;
  ui.statusMessage.textContent = text;
  document.body.classList.toggle("error", error);
}

// ============================================================================
// DEVICE CHANGES (camera plugged in / unplugged)
// ============================================================================

if (navigator.mediaDevices?.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    refreshCameraList().catch(() => {});
  });
}

// ============================================================================
// VISIBILITY CHANGE HANDLER (reacquire camera on tab switch)
// ============================================================================

document.addEventListener("visibilitychange", () => {
  // Release the camera while backgrounded — holding it would keep the
  // recording indicator lit on a page the user has navigated away from.
  if (document.hidden && state.cameraActive) {
    console.log("Tab hidden, stopping camera");
    stopCamera();
    return;
  }

  // Coming back: restore whatever the page had before, scanning or not.
  if (!document.hidden && state.cameraWanted && !state.cameraActive) {
    console.log("Tab visible, reacquiring camera");
    requestCamera(state.activeDeviceId).catch((err) => {
      console.error("Failed to reacquire camera:", err);
      if (state.scanning) stopScanning();
    });
  }
});

// ============================================================================
// STARTUP
// ============================================================================

window.addEventListener("DOMContentLoaded", initApp);
