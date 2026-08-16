// MediaPipe is imported dynamically inside loadModel() so that a CDN failure
// degrades to "camera works, detection doesn't" instead of killing the module.
const MEDIAPIPE_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8";

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
  maxObjectsPerCycle: 4,
  heartbeatInterval: 5000, // repeat closest object every 5s if nothing changed (ms)
  veryCloseThreshold: 0.7, // interrupt speech if object closer than this
  distanceBucketSize: 0.5, // only re-announce if distance changes by at least this much

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
  userGestured: false, // speech synthesis is blocked before the first interaction
  videoElement: null,
  detector: null,
  audioContext: null,
  lastAnnouncements: new Map(), // Track {distance, time} per class for change detection and heartbeat
  detectionStartTime: 0,
  lastDetections: [], // For the debug overlay
  cameras: [], // videoinput devices from enumerateDevices()
  activeDeviceId: null,
};

// ============================================================================
// UI ELEMENTS
// ============================================================================

const ui = {
  toggleButton: document.getElementById("toggle-button"),
  statusMessage: document.getElementById("status-message"),
  videoElement: document.getElementById("camera-feed"),
  canvas: document.getElementById("detection-canvas"),
  noSignal: document.getElementById("no-signal"),
  cameraDot: document.getElementById("camera-dot"),
  cameraLabel: document.getElementById("camera-label"),
  cameraMeta: document.getElementById("camera-meta"),
  switchCamera: document.getElementById("switch-camera"),
};

// ============================================================================
// INIT
// ============================================================================

async function initApp() {
  ui.toggleButton.addEventListener("click", toggleScanning);
  ui.switchCamera.addEventListener("click", switchCamera);
  state.videoElement = ui.videoElement;

  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraStatus("error", "getUserMedia unavailable", "Needs HTTPS or localhost");
    updateStatus("Camera API unavailable. Serve this page over HTTPS or localhost.");
    ui.toggleButton.disabled = true;
    return;
  }

  // Automatically request camera access on page load
  updateStatus("Requesting camera access...");
  try {
    await requestCamera();
    updateStatus("Camera ready. Tap to start scanning");
  } catch (err) {
    console.error("Camera request failed:", err);
    ui.toggleButton.disabled = true;
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
  ui.switchCamera.hidden = state.cameras.length < 2;
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
  updateStatus("Requesting camera access...");
  setCameraStatus("pending", "Requesting camera access…");

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
    setCameraStatus("error", errorMsg, lastErr?.name || "");
    updateStatus(errorMsg);
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
  ui.noSignal.hidden = true;

  const label = track?.label || "Camera";
  const facing = settings.facingMode ? ` · ${settings.facingMode}` : "";
  const res = `${state.videoElement.videoWidth}×${state.videoElement.videoHeight}`;
  const fps = settings.frameRate ? ` @ ${Math.round(settings.frameRate)}fps` : "";
  setCameraStatus(
    "live",
    label,
    `${res}${fps}${facing} · ${state.cameras.length} camera${state.cameras.length === 1 ? "" : "s"} found`
  );

  console.log("Camera active:", { label, settings, resolution: res });

  updateStatus("Camera active");
  speak("Camera active");
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

async function switchCamera() {
  if (state.cameras.length < 2) return;
  const idx = state.cameras.findIndex((c) => c.deviceId === state.activeDeviceId);
  const next = state.cameras[(idx + 1) % state.cameras.length];
  stopCamera();
  try {
    await requestCamera(next.deviceId);
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
  ui.noSignal.hidden = false;
  clearOverlay();
  setCameraStatus("idle", "Camera stopped");
  updateStatus("Camera stopped");
}

function setCameraStatus(status, label, meta = "") {
  ui.cameraDot.className = `camera-dot ${status}`;
  ui.cameraLabel.textContent = label;
  ui.cameraMeta.textContent = meta;
}

// ============================================================================
// MODEL LOADING
// ============================================================================

async function loadModel() {
  if (state.modelLoaded) return;

  try {
    updateStatus("Loading AI model...");
    speak("Loading model");

    // Note: the bundle is published as .mjs — .js 404s on the CDN.
    const { FilesetResolver, ObjectDetector } = await import(
      `${MEDIAPIPE_CDN}/vision_bundle.mjs`
    );

    const vision = await FilesetResolver.forVisionTasks(`${MEDIAPIPE_CDN}/wasm`);

    const modelAsset = `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite`;

    state.detector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelAsset,
        delegate: "GPU", // Use GPU if available, fallback to CPU
      },
      scoreThreshold: CONFIG.minConfidence,
      maxResults: 10,
      runningMode: "VIDEO",
    });

    state.modelLoaded = true;
    updateStatus("Model loaded");
    speak("Model ready");
  } catch (err) {
    console.error("Model load failed:", err);
    const errorMsg = "Model failed to load. Check your internet connection.";
    updateStatus(errorMsg);
    speak(errorMsg);
    throw err;
  }
}

// ============================================================================
// DETECTION & ANNOUNCEMENT LOGIC
// ============================================================================

async function startScanning() {
  try {
    // Initialize audio context after user gesture
    state.userGestured = true;
    initAudioContext();

    // Only request camera if not already active
    if (!state.cameraActive) {
      await requestCamera(state.activeDeviceId);
    }

    await loadModel();

    state.scanning = true;
    state.detectionStartTime = Date.now();
    ui.toggleButton.classList.add("scanning");
    ui.toggleButton.querySelector(".button-text").textContent = "TAP TO STOP";

    updateStatus("Scanning...");
    speak("Scanning started");

    detectionLoop();
  } catch (err) {
    console.error("Failed to start scanning:", err);
    state.scanning = false;
    ui.toggleButton.classList.remove("scanning");
    ui.toggleButton.querySelector(".button-text").textContent = "TAP TO START";
  }
}

function stopScanning() {
  state.scanning = false;
  state.lastAnnouncements.clear();
  stopCamera();
  speechSynthesis.cancel();
  ui.toggleButton.classList.remove("scanning");
  ui.toggleButton.querySelector(".button-text").textContent = "TAP TO START";
  updateStatus("Tap to start");
  speak("Scanning stopped");
}

function toggleScanning() {
  state.userGestured = true;
  if (state.scanning) {
    stopScanning();
  } else {
    startScanning();
  }
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
      const detections = state.detector.detectForVideo(state.videoElement, now);

      if (detections.detections && detections.detections.length > 0) {
        state.lastDetections = detections.detections;
        drawOverlay(detections.detections);
        await processDetections(detections.detections);
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

async function processDetections(detections) {
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

  // Check for very close object (interrupt)
  const veryClose = objectsWithData.find((o) => o.distance < CONFIG.veryCloseThreshold);
  if (veryClose) {
    speechSynthesis.cancel();
    const warning = `${veryClose.label} very close ahead`;
    updateStatus(warning);
    speak(warning, { rate: 1.2, pitch: 1.1 });
    playPanningTone(veryClose.centerXNorm);
    return; // Don't process further
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
    if (Math.abs(obj.distance - lastRecord.distance) >= CONFIG.distanceBucketSize)
      return true; // Distance changed
    if (now - lastRecord.time >= CONFIG.heartbeatInterval)
      return true; // Periodic repeat per object

    return false;
  });

  // Announce
  for (const obj of toAnnounce) {
    const msg = formatAnnouncement(obj);
    updateStatus(msg);
    speak(msg);
    playPanningTone(obj.centerXNorm);
    // Track distance and announcement time for per-object heartbeat
    state.lastAnnouncements.set(obj.label, { distance: obj.distance, time: now });

    // Stagger announcements slightly
    await sleep(200);
  }
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

    ctx.strokeStyle = "#4ade80";
    ctx.strokeRect(box.x, box.y, box.w, box.h);

    const metrics = ctx.measureText(text);
    const textHeight = parseInt(ctx.font, 10) * 1.3;
    ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
    ctx.fillRect(box.x, Math.max(0, box.y - textHeight), metrics.width + 10, textHeight);
    ctx.fillStyle = "#4ade80";
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
  // Browsers reject speech before the first user interaction; skip it silently
  // rather than filling the console with autoplay warnings on page load.
  if (!state.userGestured) return;

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = options.rate || 1.0;
  utterance.pitch = options.pitch || 1.0;
  utterance.volume = 1.0;

  speechSynthesis.speak(utterance);
}

function playPanningTone(centerX) {
  if (!state.audioContext) return;

  const panValue = (centerX - 0.5) * 2; // -1 (left) to +1 (right)

  try {
    const now = state.audioContext.currentTime;
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

function updateStatus(text) {
  ui.statusMessage.textContent = text;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
  if (document.hidden && state.cameraActive && state.scanning) {
    console.log("Tab hidden, stopping camera");
    stopCamera();
  } else if (!document.hidden && state.scanning && !state.cameraActive) {
    console.log("Tab visible, reacquiring camera");
    requestCamera(state.activeDeviceId).catch((err) => {
      console.error("Failed to reacquire camera:", err);
      stopScanning();
    });
  }
});

// ============================================================================
// STARTUP
// ============================================================================

window.addEventListener("DOMContentLoaded", initApp);
