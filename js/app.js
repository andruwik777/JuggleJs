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
    const d = b.height;
    const centerX = b.originX + b.width / 2;
    const centerY = b.originY + b.height / 2;
    const t = Date.now();
    pushBallState(centerX, centerY, d, true, t);
    tryCountJuggle();
    ballHighlighter.style.left = (video.offsetWidth - centerX - d / 2) + 'px';
    ballHighlighter.style.top = (centerY - d / 2) + 'px';
    ballHighlighter.style.width = d + 'px';
    ballHighlighter.style.height = d + 'px';
    ballHighlighter.style.display = 'block';
  } else {
    ballHighlighter.style.display = 'none';
  }
}
