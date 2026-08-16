import { FilesetResolver, ObjectDetector } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.js";

// ============================================================================
// CONFIG
// ============================================================================

const CONFIG = {
  // Detection
  minConfidence: 0.5,
  detectionInterval: 100, // ms between detection frames
  detectModel: "lite2", // "lite2" or "lite0"; fallback to lite0 if lite2 exceeds latency budget

  // Distance estimation
  focalLengthPx: 400, // approximate focal length for a typical phone camera
  objectSizes: {
    // Average real-world sizes per COCO class (in meters)
    person: 1.7,
    chair: 0.9,
    car: 1.8,
    bus: 3.0,
    truck: 2.5,
    dog: 0.5,
    cat: 0.25,
    bottle: 0.25,
    cup: 0.08,
    laptop: 0.35,
    phone: 0.15,
    book: 0.20,
    backpack: 0.35,
    bicycle: 1.5,
    motorcycle: 1.8,
    clock: 0.20,
    vase: 0.25,
    teddy bear: 0.4,
    sports ball: 0.24,
    frisbee: 0.27,
    skis: 1.65,
    snowboard: 1.5,
    sports equipment: 0.3,
    kite: 0.7,
    baseball bat: 0.85,
    baseball glove: 0.25,
    skateboard: 0.8,
    surfboard: 1.8,
    tennis racket: 0.68,
    bottle: 0.25,
    wine glass: 0.15,
    cup: 0.08,
    fork: 0.20,
    knife: 0.25,
    spoon: 0.20,
    bowl: 0.25,
    banana: 0.19,
    apple: 0.08,
    sandwich: 0.15,
    orange: 0.08,
    broccoli: 0.15,
    carrot: 0.20,
    hot dog: 0.15,
    pizza: 0.30,
    donut: 0.08,
    cake: 0.20,
    chair: 0.9,
    couch: 2.0,
    potted plant: 0.5,
    bed: 1.6,
    dining table: 1.5,
    toilet: 0.4,
    tv: 0.6,
    laptop: 0.35,
    mouse: 0.08,
    remote: 0.15,
    keyboard: 0.45,
    microwave: 0.5,
    oven: 0.7,
    toaster: 0.25,
    sink: 0.8,
    refrigerator: 1.7,
    book: 0.20,
    clock: 0.20,
    vase: 0.25,
    scissors: 0.20,
    teddy bear: 0.4,
    hair drier: 0.25,
    toothbrush: 0.20,
    default: 0.5, // fallback for unmapped classes
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
  videoElement: null,
  detector: null,
  audioContext: null,
  lastAnnouncements: new Map(), // Track last announced distance per class for change detection
  lastHeartbeatTime: Date.now(),
  detectionStartTime: 0,
  lastDetections: [], // For animation/overlay
};

// ============================================================================
// UI ELEMENTS
// ============================================================================

const ui = {
  toggleButton: document.getElementById("toggle-button"),
  statusMessage: document.getElementById("status-message"),
  videoElement: document.getElementById("camera-feed"),
  canvas: document.getElementById("detection-canvas"),
};

// ============================================================================
// INIT
// ============================================================================

async function initApp() {
  ui.toggleButton.addEventListener("click", toggleScanning);
  state.videoElement = ui.videoElement;

  // Initialize Web Audio API for panning cues
  try {
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
  } catch (e) {
    console.warn("Web Audio API not available:", e);
  }

  updateStatus("Tap to start");
}

// ============================================================================
// CAMERA ACCESS
// ============================================================================

async function requestCamera() {
  try {
    updateStatus("Requesting camera access...");
    const constraints = {
      video: {
        facingMode: "environment", // Rear camera
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.videoElement.srcObject = stream;

    return new Promise((resolve) => {
      state.videoElement.onloadedmetadata = () => {
        state.cameraActive = true;
        updateStatus("Camera active");
        speak("Camera active, loading model");
        resolve();
      };
    });
  } catch (err) {
    const errorMsg = err.name === "NotAllowedError"
      ? "Camera permission denied. Please allow camera access."
      : "Camera not available. Check your device.";
    updateStatus(errorMsg);
    speak(errorMsg);
    throw err;
  }
}

function stopCamera() {
  if (state.videoElement.srcObject) {
    state.videoElement.srcObject.getTracks().forEach((track) => track.stop());
  }
  state.cameraActive = false;
  updateStatus("Camera stopped");
}

// ============================================================================
// MODEL LOADING
// ============================================================================

async function loadModel() {
  if (state.modelLoaded) return;

  try {
    updateStatus("Loading AI model...");
    speak("Loading model");

    const wasmPath = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm`;

    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    const modelAsset = `https://storage.googleapis.com/mediapipe-models/object_detector/${CONFIG.detectModel}/object_detector.tflite`;

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
    await requestCamera();
    await loadModel();

    state.scanning = true;
    state.detectionStartTime = Date.now();
    state.lastHeartbeatTime = Date.now();
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
        await processDetections(detections.detections);
      } else {
        state.lastDetections = [];
      }
    } catch (err) {
      console.error("Detection error:", err);
    }
  }

  // Schedule next detection
  setTimeout(() => detectionLoop(), CONFIG.detectionInterval);
}

async function processDetections(detections) {
  const now = Date.now();

  // Compute distances and urgency tiers
  const objectsWithData = detections
    .map((d) => {
      const bbox = d.boundingBox;
      const label = d.categories[0]?.categoryName || "unknown";
      const confidence = d.categories[0]?.score || 0;

      const bboxWidth = bbox.width * state.videoElement.videoWidth;
      const bboxHeight = bbox.height * state.videoElement.videoHeight;
      const bboxAvgSize = (bboxWidth + bboxHeight) / 2;

      const realSize = CONFIG.objectSizes[label] || CONFIG.objectSizes.default;
      const distance = (realSize * CONFIG.focalLengthPx) / bboxAvgSize;

      const centerX = bbox.originX + bbox.width / 2;
      const direction = centerX < 0.33 ? "left" : centerX > 0.67 ? "right" : "center";

      let tier = "far";
      if (distance < CONFIG.urgencyTiers.critical) tier = "critical";
      else if (distance < CONFIG.urgencyTiers.near) tier = "near";

      return {
        label,
        distance,
        confidence,
        direction,
        tier,
        centerX,
        bbox,
      };
    })
    .sort((a, b) => a.distance - b.distance); // Sort by distance

  // Check for very close object (interrupt)
  const veryClose = objectsWithData.find((o) => o.distance < CONFIG.veryCloseThreshold);
  if (veryClose) {
    speechSynthesis.cancel();
    const warning = `${veryClose.label} very close ahead`;
    updateStatus(warning);
    speak(warning, { rate: 1.2, pitch: 1.1 });
    playPanningTone(veryClose.centerX);
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
    const key = obj.label;
    const lastDist = state.lastAnnouncements.get(key);

    if (lastDist === undefined) return true; // New object
    if (Math.abs(obj.distance - lastDist) >= CONFIG.distanceBucketSize)
      return true; // Distance changed
    if (now - state.lastHeartbeatTime >= CONFIG.heartbeatInterval)
      return true; // Periodic repeat

    return false;
  });

  // Announce
  if (toAnnounce.length > 0) {
    state.lastHeartbeatTime = now;
    for (const obj of toAnnounce) {
      const msg = formatAnnouncement(obj);
      updateStatus(msg);
      speak(msg);
      playPanningTone(obj.centerX);
      state.lastAnnouncements.set(obj.label, obj.distance);

      // Stagger announcements slightly
      await sleep(200);
    }
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
// VISIBILITY CHANGE HANDLER (reacquire camera on tab switch)
// ============================================================================

document.addEventListener("visibilitychange", () => {
  if (document.hidden && state.cameraActive) {
    console.log("Tab hidden, stopping camera");
    stopCamera();
  } else if (!document.hidden && state.scanning && !state.cameraActive) {
    console.log("Tab visible, reacquiring camera");
    requestCamera().catch((err) => {
      console.error("Failed to reacquire camera:", err);
      stopScanning();
    });
  }
});

// ============================================================================
// STARTUP
// ============================================================================

window.addEventListener("DOMContentLoaded", initApp);
