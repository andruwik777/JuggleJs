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
  window.dispatchEvent(new Event('juggleAppReady'));
};
initializeObjectDetector();

let video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const videoStage = document.getElementById('videoStage');
const aiMsEl = document.getElementById('aiMs');
const postAiMsEl = document.getElementById('postAiMs');
const totalMsFpsEl = document.getElementById('totalMsFps');
const juggleCountEl = document.getElementById('juggleCount');
let enableWebcamButton;

const STATE_BUFFER_CAPACITY = 30;
let juggleCount = 0;
let ballState = [];
let lastLocalMinY = null;

/**
 * 1D Kalman filter, state [position, velocity, acceleration]. Same model as JuggleNet (KalmanFilter.py).
 * F = [[1, dt, 0.5*dt^2], [0, 1, dt], [0, 0, 1]], H = [1, 0, 0], we only observe position.
 */
function Kalman1D(processVariance, measurementVariance) {
  this.x = [0, 0, 0];
  this.P = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  this.H = [1, 0, 0];
  this.R = measurementVariance;
  this.q = processVariance;
  this.initialised = false;
}

Kalman1D.prototype.setF = function (dt) {
  const dt2 = 0.5 * dt * dt;
  this.F = [1, dt, dt2, 0, 1, dt, 0, 0, 1];
};

Kalman1D.prototype.update = function (z) {
  const Hx = this.H[0] * this.x[0] + this.H[1] * this.x[1] + this.H[2] * this.x[2];
  const y = z - Hx;
  const HP = [this.P[0], this.P[1], this.P[2]];
  const S = HP[0] + this.R;
  const K = [this.P[0] / S, this.P[3] / S, this.P[6] / S];
  this.x[0] += K[0] * y;
  this.x[1] += K[1] * y;
  this.x[2] += K[2] * y;
  const oneMinusK0 = 1 - K[0];
  this.P[0] = oneMinusK0 * this.P[0];
  this.P[1] = oneMinusK0 * this.P[1];
  this.P[2] = oneMinusK0 * this.P[2];
  this.P[3] = -K[1] * this.P[0] + this.P[3];
  this.P[4] = -K[1] * this.P[1] + this.P[4];
  this.P[5] = -K[1] * this.P[2] + this.P[5];
  this.P[6] = -K[2] * this.P[0] + this.P[6];
  this.P[7] = -K[2] * this.P[1] + this.P[7];
  this.P[8] = -K[2] * this.P[2] + this.P[8];
  this.initialised = true;
};

Kalman1D.prototype.predict = function (dt) {
  if (dt <= 0) return this.x[0];
  this.setF(dt);
  const F = this.F;
  this.x = [
    F[0] * this.x[0] + F[1] * this.x[1] + F[2] * this.x[2],
    F[3] * this.x[0] + F[4] * this.x[1] + F[5] * this.x[2],
    F[6] * this.x[0] + F[7] * this.x[1] + F[8] * this.x[2]
  ];
  const P = this.P;
  const FP = [
    F[0] * P[0] + F[1] * P[3] + F[2] * P[6], F[0] * P[1] + F[1] * P[4] + F[2] * P[7], F[0] * P[2] + F[1] * P[5] + F[2] * P[8],
    F[3] * P[0] + F[4] * P[3] + F[5] * P[6], F[3] * P[1] + F[4] * P[4] + F[5] * P[7], F[3] * P[2] + F[4] * P[5] + F[5] * P[8],
    F[6] * P[0] + F[7] * P[3] + F[8] * P[6], F[6] * P[1] + F[7] * P[4] + F[8] * P[7], F[6] * P[2] + F[7] * P[5] + F[8] * P[8]
  ];
  this.P = [
    FP[0] * F[0] + FP[1] * F[1] + FP[2] * F[2] + this.q, FP[0] * F[3] + FP[1] * F[4] + FP[2] * F[5], FP[0] * F[6] + FP[1] * F[7] + FP[2] * F[8],
    FP[3] * F[0] + FP[4] * F[1] + FP[5] * F[2], FP[3] * F[3] + FP[4] * F[4] + FP[5] * F[5] + this.q, FP[3] * F[6] + FP[4] * F[7] + FP[5] * F[8],
    FP[6] * F[0] + FP[7] * F[1] + FP[8] * F[2], FP[6] * F[3] + FP[7] * F[4] + FP[8] * F[5], FP[6] * F[6] + FP[7] * F[7] + FP[8] * F[8] + this.q
  ];
  return this.x[0];
};

/** Ball: two 1D filters (X and Y). */
const KALMAN_PROCESS_VARIANCE = 0.01;
const KALMAN_MEASUREMENT_VARIANCE = 0.1;
let kfBallX = null;
let kfBallY = null;
let lastKalmanT = null;

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

let ballHighlighter = null;

/** Snake visualization: frame + dot elements, created once and reused. */
let snakeFrame = null;
let snakeDots = [];
const SNAKE_DOT_SIZE = 5;
const SNAKE_DOT_SIZE_JUGGLE = 10;

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
  const wrap = document.getElementById('webcamButtonWrap');
  if (wrap) wrap.classList.add('removed');

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
let rafId = null;

/**
 * Run one frame of detection (used by frame-driven video test only).
 * Set DevTools breakpoint on the detectForVideo line below to pause at this frame; video will stay on this frame.
 */
async function runOneDetectionFrame() {
  if (runningMode === 'IMAGE') {
    runningMode = 'VIDEO';
    await objectDetector.setOptions({ runningMode: 'VIDEO' });
  }
  const startTimeMs = performance.now();
  lastVideoTime = video.currentTime;
  const detections = objectDetector.detectForVideo(video, startTimeMs); // breakpoint here in test
  displayVideoDetections(detections);
}

async function predictWebcam() {
  if (video.ended) {
    rafId = null;
    return;
  }
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
  if (hadNewFrame && aiMsEl && postAiMsEl && totalMsFpsEl) {
    const postAiMs = predictWebcamMs - detectForVideoMs;
    const fps = predictWebcamMs > 0 ? 1000 / predictWebcamMs : 0;
    aiMsEl.textContent = 'AI: ' + detectForVideoMs + ' ms';
    postAiMsEl.textContent = 'PostAI: ' + postAiMs + ' ms';
    totalMsFpsEl.textContent = 'Total: ' + predictWebcamMs + ' ms / ' + fps.toFixed(1) + ' FPS';
  }
  rafId = window.requestAnimationFrame(predictWebcam);
}

function setJuggleCount(n) {
  juggleCount = n;
  if (juggleCountEl) juggleCountEl.textContent = n + ' juggles';
}

/**
 * Append one point to the ball trajectory state (used for snake and juggle counting).
 * @param {number} x - X position (display)
 * @param {number} y - Y position (display)
 * @param {number} d - Diameter / size for display
 * @param {boolean} calculatedOnly - True if from Kalman predict only (no detection)
 * @param {number} t - Timestamp (ms)
 * @param {number} [vx] - Optional velocity X (otherwise derived from previous point)
 * @param {number} [vy] - Optional velocity Y (otherwise derived from previous point)
 * @param {number|null} [juggleCount=null] - Ordinal juggle number when this frame is a counted juggle peak
 * @param {{ line1: string, line2: string }|null} [text=null] - Debug label (line1 e.g. "7" or "-", line2 e.g. "318,2.1")
 */
function pushBallState(x, y, d, calculatedOnly, t, vx, vy, juggleCount = null, text = null) {
  let vxOut = vx != null ? vx : 0;
  let vyOut = vy != null ? vy : 0;
  if (ballState.length > 0 && vxOut === 0 && vyOut === 0) {
    const prev = ballState[ballState.length - 1];
    const dtSec = (t - prev.t) / 1000;
    if (dtSec > 0) {
      vxOut = (x - prev.x) / dtSec;
      vyOut = (y - prev.y) / dtSec;
    }
  }
  ballState.push({ x, y, vx: vxOut, vy: vyOut, d, calculatedOnly, t, juggleCount: juggleCount ?? null, text: text ?? null });
  if (ballState.length > STATE_BUFFER_CAPACITY) ballState.shift();
}

/**
 * Check if the latest detected point forms a local max (peak); optionally whether it counts as a juggle.
 * Uses only non-calculated points.
 * @returns {{ isJuggleDetected: boolean, ratio: number|null }} ratio = dropFromTop/minAmplitude (1 decimal), set only at peaks
 */
function isNewJuggleDetected() {
  const detected = ballState.filter((e) => !e.calculatedOnly);
  if (detected.length < 3) return { isJuggleDetected: false, ratio: null };
  const n = detected.length;
  const prev = detected[n - 2];
  const curr = detected[n - 1];
  const prevPrev = detected[n - 3];
  if (prev.y <= prevPrev.y && prev.y <= curr.y) {
    lastLocalMinY = prev.y;
  }
  if (prev.y >= prevPrev.y && prev.y >= curr.y) {
    const dropFromTop = prev.y - (lastLocalMinY != null ? lastLocalMinY : prev.y);
    const minAmplitude = prev.d / 2;
    const ratio = minAmplitude > 0 ? Math.round((dropFromTop / minAmplitude) * 10) / 10 : 0;
    const isJuggleDetected = dropFromTop >= minAmplitude;
    return { isJuggleDetected, ratio };
  }
  return { isJuggleDetected: false, ratio: null };
}

/**
 * Update the peak point (second-to-last in ballState) with juggleCount and text from isNewJuggleDetected result.
 * @param {{ isJuggleDetected: boolean, ratio: number|null }} result
 */
function setJuggleInBallState(result) {
  if (result.ratio == null || ballState.length < 2) return;
  const peak = ballState[ballState.length - 2];
  if (result.isJuggleDetected) {
    juggleCount++;
    setJuggleCount(juggleCount);
    peak.juggleCount = juggleCount;
  } else {
    peak.juggleCount = null;
  }
  const line1 = peak.juggleCount != null ? String(peak.juggleCount) : '-';
  const line2 = Math.round(peak.y * 10) / 10 + ',' + result.ratio;
  peak.text = { line1, line2 };
}

function displayVideoDetections(result) {
  const container = videoStage || liveView;
  if (!ballHighlighter) {
    ballHighlighter = document.createElement('div');
    ballHighlighter.setAttribute('class', 'highlighter');
    container.appendChild(ballHighlighter);
  }
  const t = Date.now();
  const dtSec = lastKalmanT != null ? (t - lastKalmanT) / 1000 : 0;
  lastKalmanT = t;

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

    if (!kfBallX) {
      kfBallX = new Kalman1D(KALMAN_PROCESS_VARIANCE, KALMAN_MEASUREMENT_VARIANCE);
      kfBallY = new Kalman1D(KALMAN_PROCESS_VARIANCE, KALMAN_MEASUREMENT_VARIANCE);
    }
    if (!kfBallX.initialised) {
      kfBallX.x[0] = centerXDisplay;
      kfBallX.x[1] = 0;
      kfBallX.x[2] = 0;
      kfBallX.initialised = true;
    }
    if (!kfBallY.initialised) {
      kfBallY.x[0] = centerYDisplay;
      kfBallY.x[1] = 0;
      kfBallY.x[2] = 0;
      kfBallY.initialised = true;
    }
    kfBallX.update(centerXDisplay);
    kfBallY.update(centerYDisplay);
    const smoothedX = kfBallX.x[0];
    const smoothedY = kfBallY.x[0];
    const vx = kfBallX.x[1];
    const vy = kfBallY.x[1];
    kfBallX.predict(dtSec);
    kfBallY.predict(dtSec);

    pushBallState(smoothedX, smoothedY, dDisplay, false, t, vx, vy, null, null);
    const juggleResult = isNewJuggleDetected();
    if (juggleResult.ratio != null) setJuggleInBallState(juggleResult);

    ballHighlighter.style.left = (dw - centerXDisplay - dDisplay / 2) + 'px';
    ballHighlighter.style.top = (centerYDisplay - dDisplay / 2) + 'px';
    ballHighlighter.style.width = dDisplay + 'px';
    ballHighlighter.style.height = dDisplay + 'px';
    ballHighlighter.style.display = 'block';
  } else {
    ballHighlighter.style.display = 'none';
    if (kfBallX && kfBallY && kfBallX.initialised) {
      const predX = kfBallX.predict(dtSec);
      const predY = kfBallY.predict(dtSec);
      const d = ballState.length > 0 ? ballState[ballState.length - 1].d : 40;
      pushBallState(predX, predY, d, true, t, undefined, undefined, null, null);
    }
  }
  liveSnakeVisualisation();
}

/**
 * Draw ballState as a "snake" in a frame above the timing stats (AI/PostAI/Total), 5px gap.
 * Same width as stats block (videoStage), oldest left, newest right; Y scaled to frame height.
 */
function liveSnakeVisualisation() {
  const n = ballState.length;
  if (n === 0) {
    if (snakeFrame) snakeFrame.style.display = 'none';
    return;
  }

  if (!snakeFrame) {
    snakeFrame = document.createElement('div');
    snakeFrame.setAttribute('class', 'snake-frame');
    videoStage.appendChild(snakeFrame);
  }
  snakeFrame.style.display = 'block';

  const frameW = snakeFrame.offsetWidth || videoStage.offsetWidth || 300;
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
    const dotSize = pt.juggleCount != null ? SNAKE_DOT_SIZE_JUGGLE : SNAKE_DOT_SIZE;
    const half = dotSize / 2;
    const xFrac = n > 1 ? i / (n - 1) : 0.5;
    const x = xFrac * frameW;
    const yFrac = rangeY > 0 ? (pt.y - minY) * yScale : 0.5;
    const y = yFrac * frameH;
    const el = snakeDots[i];
    el.style.left = (x - half) + 'px';
    el.style.top = (y - half) + 'px';
    el.style.width = dotSize + 'px';
    el.style.height = dotSize + 'px';
    el.style.display = 'block';
    if (pt.juggleCount != null) {
      el.classList.add('snake-dot-juggle');
    } else {
      el.classList.remove('snake-dot-juggle');
    }
    if (pt.calculatedOnly) {
      el.classList.add('snake-dot-calculated');
    } else {
      el.classList.remove('snake-dot-calculated');
    }
    if (pt.text) {
      let label = el.querySelector('.snake-dot-label');
      if (!label) {
        label = document.createElement('div');
        label.setAttribute('class', 'snake-dot-label');
        el.appendChild(label);
      }
      label.innerHTML = pt.text.line1 + '<br>' + pt.text.line2;
      label.style.display = 'block';
    } else {
      const label = el.querySelector('.snake-dot-label');
      if (label) label.style.display = 'none';
    }
  }
  for (let i = n; i < snakeDots.length; i++) {
    snakeDots[i].style.display = 'none';
  }
}

function resetJuggleState() {
  juggleCount = 0;
  ballState.length = 0;
  lastLocalMinY = null;
  kfBallX = null;
  kfBallY = null;
  lastKalmanT = null;
  lastVideoTime = -1;
  if (rafId != null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (juggleCountEl) juggleCountEl.textContent = '0 juggles';
}

/**
 * Run detection on a video file. Returns { start, result } so the test page can call start() from a user
 * click (required by browser autoplay policy). When the debugger is paused in the loop, the video stops.
 * @param {string} videoUrl - URL or path to the MP4 file (e.g. 'test.mp4')
 * @returns {{ start: (debugMode?: boolean) => void, result: Promise<number> }} - start(debugMode) from user click; debugMode true = frame-driven (breakpoints work, ~0.5x); false = real-time playback
 */
window.runJuggleTest = function (videoUrl) {
  let resolveResult;
  let rejectResult;
  const result = new Promise((resolve, reject) => { resolveResult = resolve; rejectResult = reject; });
  if (!objectDetector) {
    return { start: () => {}, result: Promise.reject(new Error('Detector not ready')) };
  }
  resetJuggleState();
  document.body.classList.add('live-active');
  liveView.classList.add('live-fullscreen');
  const wrap = document.getElementById('webcamButtonWrap');
  if (wrap) wrap.classList.add('removed');
  if (juggleCountEl) juggleCountEl.classList.remove('hidden');

  video.src = videoUrl;
  video.load();
  video.addEventListener('loadeddata', function onLoaded() {
    video.removeEventListener('loadeddata', onLoaded);
    resizeStageToContain();
    window.addEventListener('resize', resizeStageToContain);
    const TEST_FPS = 30;

    function runDebugMode() {
      let nextTime = 0;
      function step() {
        if (nextTime >= video.duration) {
          if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
          resolveResult(juggleCount);
          return;
        }
        video.currentTime = nextTime;
        video.addEventListener('seeked', function onSeeked() {
          video.removeEventListener('seeked', onSeeked);
          runOneDetectionFrame().then(() => {
            nextTime += 1 / TEST_FPS;
            rafId = requestAnimationFrame(step);
          });
        }, { once: true });
      }
      step();
    }

    function runRealtimeMode() {
      let lastProcessedFrame = -1;
      function step() {
        if (video.ended) {
          if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
          resolveResult(juggleCount);
          return;
        }
        const currentFrame = Math.floor(video.currentTime * TEST_FPS);
        if (currentFrame > lastProcessedFrame) {
          lastProcessedFrame = currentFrame;
          runOneDetectionFrame().then(() => {
            rafId = requestAnimationFrame(step);
          });
        } else {
          rafId = requestAnimationFrame(step);
        }
      }
      video.play().then(() => { rafId = requestAnimationFrame(step); }).catch((e) => rejectResult(e));
    }

    window.runJuggleTestStart = function start(debugMode) {
      window.runJuggleTestStart = null;
      if (debugMode) runDebugMode();
      else runRealtimeMode();
    };
    window.dispatchEvent(new Event('juggleTestReadyToRun'));
  }, { once: true });
  video.addEventListener('error', function onError() {
    video.removeEventListener('error', onError);
    if (rejectResult) rejectResult(new Error('Video failed to load'));
  }, { once: true });
  return {
    start: function (debugMode) { if (window.runJuggleTestStart) window.runJuggleTestStart(debugMode); },
    result
  };
};
