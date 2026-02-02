// Based on CodePen: https://codepen.io/mediapipe-preview/pen/vYrWvNg
// Guide: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/web_js
import { ObjectDetector, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/vision_bundle.mjs';

const demosSection = document.getElementById('demos');

let objectDetector;
let runningMode = 'IMAGE';

const initializeObjectDetector = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm'
  );
  const MODEL_PATH = './models/model_fp16.tflite';
  const DETECTION_CATEGORY_NAME = 'Juggling - v7 2022-07-26 4-53pm';
  objectDetector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_PATH,
      delegate: 'GPU'
    },
    scoreThreshold: 0.4,
    maxResults: 1,
    runningMode: runningMode,
    categoryAllowlist: [DETECTION_CATEGORY_NAME]
  });
  demosSection.classList.remove('invisible');
};
initializeObjectDetector();

let video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const videoStage = document.getElementById('videoStage');
const detectForVideoMsEl = document.getElementById('detectForVideoMs');
const predictWebcamMsEl = document.getElementById('predictWebcamMs');
const deltaMsEl = document.getElementById('deltaMs');
const juggleCountEl = document.getElementById('juggleCount');
let enableWebcamButton;

const STATE_BUFFER_CAPACITY = 30;
let juggleCount = 0;
let ballState = [];
let lastLocalMinY = null;

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

let ballHighlighter = null;

/** Snake visualization: frame + dot elements, created once and reused. */
let snakeFrame = null;
let snakeDots = [];
const SNAKE_DOT_SIZE = 5;

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById('webcamButton');
  enableWebcamButton.addEventListener('click', enableCam);
} else {
  console.warn('getUserMedia() is not supported by your browser');
}

async function enableCam(event) {
  if (!objectDetector) {
    console.log('Wait! objectDetector not loaded yet.');
    return;
  }

  enableWebcamButton.classList.add('removed');

  const constraints = { video: true };

  navigator.mediaDevices
    .getUserMedia(constraints)
    .then(function (stream) {
      video.srcObject = stream;
      if (juggleCountEl) juggleCountEl.classList.remove('hidden');
      document.body.classList.add('live-active');
      liveView.classList.add('live-fullscreen');
      video.addEventListener('loadeddata', onVideoReady);
    })
    .catch((err) => {
      console.error(err);
    });
}

function resizeStageToContain() {
  if (!videoStage || !video.videoWidth) return;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const r = video.videoWidth / video.videoHeight;
  let w = vw;
  let h = vw / r;
  if (h > vh) {
    h = vh;
    w = vh * r;
  }
  videoStage.style.width = w + 'px';
  videoStage.style.height = h + 'px';
}

function onVideoReady() {
  resizeStageToContain();
  window.addEventListener('resize', resizeStageToContain);
  predictWebcam();
}

let lastVideoTime = -1;
async function predictWebcam() {
  const t0 = performance.now();
  let hadNewFrame = false;
  let detectForVideoMs = 0;

  if (runningMode === 'IMAGE') {
    runningMode = 'VIDEO';
    await objectDetector.setOptions({ runningMode: 'VIDEO' });
  }
  let startTimeMs = performance.now();

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const t1 = performance.now();
    const detections = objectDetector.detectForVideo(video, startTimeMs);
    const t2 = performance.now();
    detectForVideoMs = Math.round(t2 - t1);
    hadNewFrame = true;
    displayVideoDetections(detections);
  }

  const t3 = performance.now();
  const predictWebcamMs = Math.round(t3 - t0);
  if (hadNewFrame && detectForVideoMsEl && predictWebcamMsEl && deltaMsEl) {
    const delta = predictWebcamMs - detectForVideoMs;
    detectForVideoMsEl.textContent = 'detectForVideo: ' + detectForVideoMs + ' ms';
    predictWebcamMsEl.textContent = 'predictWebcam: ' + predictWebcamMs + ' ms';
    deltaMsEl.textContent = 'DELTA: ' + delta + ' ms';
  }
  window.requestAnimationFrame(predictWebcam);
}

function setJuggleCount(n) {
  juggleCount = n;
  if (juggleCountEl) juggleCountEl.textContent = n + ' набиваний';
}

function pushBallState(x, y, d, f, t) {
  var vx = 0;
  var vy = 0;
  if (ballState.length > 0) {
    var prev = ballState[ballState.length - 1];
    var dtSec = (t - prev.t) / 1000;
    if (dtSec > 0) {
      vx = (x - prev.x) / dtSec;
      vy = (y - prev.y) / dtSec;
    }
  }
  ballState.push({ x: x, y: y, vx: vx, vy: vy, d: d, f: f, t: t });
  if (ballState.length > STATE_BUFFER_CAPACITY) ballState.shift();
}

function tryCountJuggle() {
  if (ballState.length < 3) return;
  var n = ballState.length;
  var prev = ballState[n - 2];
  var curr = ballState[n - 1];
  var prevPrev = ballState[n - 3];
  if (prev.y <= prevPrev.y && prev.y <= curr.y) {
    lastLocalMinY = prev.y;
  }
  if (prev.y >= prevPrev.y && prev.y >= curr.y) {
    var dropFromTop = prev.y - (lastLocalMinY != null ? lastLocalMinY : prev.y);
    var minAmplitude = prev.d / 2;
    if (dropFromTop >= minAmplitude) {
      setJuggleCount(juggleCount + 1);
    }
  }
}

function displayVideoDetections(result) {
  const container = videoStage || liveView;
  if (!ballHighlighter) {
    ballHighlighter = document.createElement('div');
    ballHighlighter.setAttribute('class', 'highlighter');
    container.appendChild(ballHighlighter);
  }
  const detection = result.detections && result.detections[0];
  if (detection && detection.boundingBox) {
    const b = detection.boundingBox;
    const vw = video.videoWidth || 1;
    const vh = video.videoHeight || 1;
    const dw = video.offsetWidth;
    const dh = video.offsetHeight;
    const sx = dw / vw;
    const sy = dh / vh;
    const centerX = b.originX + b.width / 2;
    const centerY = b.originY + b.height / 2;
    const centerXDisplay = centerX * sx;
    const centerYDisplay = centerY * sy;
    const dDisplay = b.height * Math.min(sx, sy);
    const t = Date.now();
    pushBallState(centerXDisplay, centerYDisplay, dDisplay, true, t);
    tryCountJuggle();
    ballHighlighter.style.left = (dw - centerXDisplay - dDisplay / 2) + 'px';
    ballHighlighter.style.top = (centerYDisplay - dDisplay / 2) + 'px';
    ballHighlighter.style.width = dDisplay + 'px';
    ballHighlighter.style.height = dDisplay + 'px';
    ballHighlighter.style.display = 'block';
  } else {
    ballHighlighter.style.display = 'none';
  }
  liveVisualisation();
}

/**
 * Draw ballState as a "snake" in a frame at bottom-left: 75vw x 20vh.
 * Oldest point left, newest right; Y scaled to frame height each frame.
 */
function liveVisualisation() {
  const n = ballState.length;
  if (n === 0) {
    if (snakeFrame) snakeFrame.style.display = 'none';
    return;
  }

  if (!snakeFrame) {
    snakeFrame = document.createElement('div');
    snakeFrame.setAttribute('class', 'snake-frame');
    liveView.appendChild(snakeFrame);
  }
  snakeFrame.style.display = 'block';

  const half = SNAKE_DOT_SIZE / 2;
  const frameW = snakeFrame.offsetWidth || Math.round(window.innerWidth * 0.75);
  const frameH = snakeFrame.offsetHeight || Math.round(window.innerHeight * 0.2);

  while (snakeDots.length < n) {
    const dot = document.createElement('div');
    dot.setAttribute('class', 'snake-dot');
    dot.setAttribute('aria-hidden', 'true');
    snakeFrame.appendChild(dot);
    snakeDots.push(dot);
  }

  let minY = ballState[0].y;
  let maxY = ballState[0].y;
  for (let i = 1; i < n; i++) {
    const y = ballState[i].y;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const rangeY = maxY - minY;
  const yScale = rangeY > 0 ? 1 / rangeY : 0;

  for (let i = 0; i < n; i++) {
    const pt = ballState[i];
    const xFrac = n > 1 ? i / (n - 1) : 0.5;
    const x = xFrac * frameW;
    const yFrac = rangeY > 0 ? (pt.y - minY) * yScale : 0.5;
    const y = (1 - yFrac) * frameH;
    const el = snakeDots[i];
    el.style.left = (x - half) + 'px';
    el.style.top = (y - half) + 'px';
    el.style.display = 'block';
  }
  for (let i = n; i < snakeDots.length; i++) {
    snakeDots[i].style.display = 'none';
  }
}
