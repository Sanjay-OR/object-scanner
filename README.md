# Object Scanner - Accessible AI Object Detection

A web-based object scanner designed for blind and low-vision users. Points a phone camera at objects and uses AI to identify them, report their distance, and announce their direction via audio.

## Features

- **Real-time object detection** using MediaPipe Tasks Vision (EfficientDet model)
- **Urgency-based audio announcements** — closest/most important objects first
- **Directional audio cues** — stereo panning tones indicate left/right position
- **Distance estimation** — approximate meters/centimeters using bounding-box heuristics
- **Accessibility-first** — fully navigable with audio only, works with screen readers
- **No backend** — entirely client-side, runs in the browser with camera access

## How to Use

1. **On mobile** (iOS or Android):
   - Open this page in your browser (Safari on iOS or Chrome on Android)
   - Tap the large blue button to start scanning
   - The app will request camera permission — grant it
   - Listen for audio announcements of nearby objects
   - Tap again to stop

2. **On desktop** (for testing):
   - Open in a modern browser (Chrome, Firefox, Safari)
   - Tap to start; if you have a webcam, it will use it
   - Visual bounding boxes are drawn for debugging

## How it Works

### Detection
- Captures video from your rear camera
- Runs each frame through the EfficientDet-Lite0 object detector
- Detects 80 common object classes (person, chair, car, dog, bottle, laptop, etc.)

### Distance Estimation
- Compares the object's bounding box **height** against a table of typical
  real-world heights (`CONFIG.objectHeights`)
- Pinhole camera math: `distance ≈ (real_height × focal_length) / bbox_height`
- Focal length is derived from the live frame width and an assumed field of
  view, so it stays correct when the capture resolution changes
- **A rough estimate, not a measurement.** It assumes the object is an average
  member of its class — a child and a tall adult are both "person", 1.7m — so
  expect it to be off, sometimes by a lot. Distances are announced as
  "about ..." and rounded to coarse steps for that reason
- An object taller than the frame is clipped, so its height is understated and
  the reported distance is too large — treat close-range readings as an upper
  bound

### Direction
- Reports object position relative to the frame: **left**, **center**, or **right**
- Also plays a stereo-panned audio tone for faster non-verbal cuing

### Smart Announcements
- **Continuous auto-scanning** with throttled announcements
- **Urgency tiers**: critical (<1.2m), near (1.2–3m), far (>3m)
- **Closest objects first** — announces only as many objects as the user needs
- **Heartbeat repeat** — re-announces unchanged objects every ~5 seconds
- **Interrupts on danger** — if something is very close (<0.7m), speaks immediately

## Technical Details

### Technology Stack
- **HTML5**: camera access via `getUserMedia`
- **MediaPipe Tasks Vision**: object detection (runs on-device)
- **Web Audio API**: stereo panning for directional cues
- **Web Speech API**: text-to-speech announcements
- **No external server** — purely client-side

### Performance
- Target latency: **<1.5 seconds** from camera frame to announcement
- EfficientDet-Lite2 inference: ~100–300ms on modern phones (GPU delegate)
- Detections throttled to ~10 per second to stay responsive

### Supported Browsers
- **iOS**: Safari 14+
- **Android**: Chrome 90+, Firefox 88+
- **Desktop**: Chrome 90+, Firefox 88+, Safari 14+
- Requires HTTPS (or localhost for testing)

## Known Limitations

- **No depth sensor** — distance is an approximation, not a measurement
- **Limited object classes** — doors, walls, stairs not in the standard COCO-80 set
- **Lighting dependent** — works best in moderate to bright conditions
- **No WiFi required** — but model loads from CDN on first use

## Calibration

The one number worth tuning per device is the camera's field of view — no
browser API reports it, so it is assumed.

1. **Measure**: stand a person (or any object in `CONFIG.objectHeights`) at a
   known distance — 3 metres is a good choice — and note what the app announces
2. **Adjust**: change `CONFIG.assumedHorizontalFovDeg` in `app.js`. A *larger*
   FOV shortens reported distances, a *smaller* one lengthens them. The default
   of 65° suits a typical phone rear camera; laptop webcams are usually 60–70°
3. **Check it holds**: repeat at a different distance. The pinhole relation is
   linear, so one FOV value should fit every distance — if it does not, the
   error is the assumed object height, not the FOV
4. **Height table**: refine `CONFIG.objectHeights` for objects you care about.
   These are *heights*, in metres, and are paired with the bounding box height

## Choosing the detection model

`CONFIG.detectModel` selects between the entries in `MODELS`:

- `lite0` (default) — EfficientDet-Lite0. Fewer confident mislabels and about
  twice as fast in local testing
- `lite2` — EfficientDet-Lite2. Higher published COCO mAP (33.97 vs 25.69), so
  it may localise better on scenes unlike the test set, which in turn feeds the
  distance estimate

## Deployment to GitHub Pages

1. **Initialize repo**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: object scanner"
   ```

2. **Create GitHub repo**:
   - Go to https://github.com/new
   - Name it (e.g., `object-scanner`)
   - Do NOT initialize with README

3. **Push**:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/object-scanner.git
   git push -u origin main
   ```

4. **Enable Pages**:
   - Go to your repo → Settings → Pages
   - Set source to `main` branch, root folder
   - Your site will be live at `https://YOUR_USERNAME.github.io/object-scanner/`

5. **Test on mobile**:
   - Visit your URL on an iPhone or Android phone
   - Tap to start and enjoy!

## Privacy

- **No data sent anywhere** — all processing happens on your device
- Camera feed never leaves your phone
- No user tracking, analytics, or ads

## Contributing & Feedback

- Report bugs or suggest features via GitHub issues
- Test on your device and share what works/doesn't

---

**Made for accessibility.** Enjoy exploring your world with audio.
