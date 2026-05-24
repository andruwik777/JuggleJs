// Based on CodePen: https://codepen.io/mediapipe-preview/pen/vYrWvNg
// Guide: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/web_js
import { ObjectDetector, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/vision_bundle.mjs';
import { Kalman1D } from './kalman1d.js';

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
  if (isLiveWebcamPage() && hasGetUserMedia()) {
    enableCam();
  } else if (isLiveWebcamPage()) {
    console.warn('getUserMedia() is not supported by your browser');
  }
};
initializeObjectDetector();

let video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const videoStage = document.getElementById('videoStage');
const aiMsEl = document.getElementById('aiMs');
const postAiMsEl = document.getElementById('postAiMs');
const totalMsFpsEl = document.getElementById('totalMsFps');
const juggleCountEl = document.getElementById('juggleCount');
const voiceCountCheckbox = document.getElementById('voiceCountCheckbox');

function isLiveWebcamPage() {
  return document.getElementById('testPanel') == null;
}

const JUGGLE_COUNT_WORDS = [
  'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten',
  'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen', 'Twenty',
];
let preferredVoice = null;

function initVoiceCount() {
  if (typeof speechSynthesis === 'undefined') return;
  const pickVoice = () => {
    const voices = speechSynthesis.getVoices();
    preferredVoice = voices.find((v) => v.lang.startsWith('en') && v.localService)
      ?? voices.find((v) => v.lang.startsWith('en'))
      ?? null;
  };
  pickVoice();
  speechSynthesis.addEventListener('voiceschanged', pickVoice);
}

function speakJuggleCount(n) {
  if (typeof speechSynthesis === 'undefined') return;
  const word = JUGGLE_COUNT_WORDS[n - 1] ?? String(n);
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(word);
  utterance.lang = 'en-US';
  utterance.rate = 1.1;
  if (preferredVoice) utterance.voice = preferredVoice;
  speechSynthesis.speak(utterance);
}

initVoiceCount();

const STATE_BUFFER_CAPACITY = Math.floor(window.innerWidth / 5);
console.log('STATE_BUFFER_CAPACITY', STATE_BUFFER_CAPACITY);
let juggleCount = 0;
let ballState = [];
let lastLocalMinY = null;

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

if (hasGetUserMedia() && isLiveWebcamPage()) {
  document.body.classList.add('live-active');
  liveView.classList.add('live-fullscreen');
}

async function enableCam() {
  if (!objectDetector) return;

  const constraints = { video: { facingMode: 'user' } };

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
  if (voiceCountCheckbox?.checked) speakJuggleCount(n);
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
 * @param {string|null} [topText=null] - Debug label above dot (e.g. "7" or "-")
 * @param {string|null} [bottomText=null] - Debug label below dot (e.g. ratio)
 */
function pushBallState(x, y, d, calculatedOnly, t, vx, vy, juggleCount = null, topText = null, bottomText = null) {
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
  ballState.push({ x, y, vx: vxOut, vy: vyOut, d, calculatedOnly, t, juggleCount: juggleCount ?? null, topText: topText ?? null, bottomText: bottomText ?? null });
  if (ballState.length > STATE_BUFFER_CAPACITY) ballState.shift();
}

/**
 * Check if the latest detected point forms a local max (peak); optionally whether it counts as a juggle.
 * Uses only non-calculated points.
 * @returns {{ isJuggleDetected: boolean, ratio: number|null }} ratio = dropFromTop/diameter (1 decimal), set only at peaks
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
    const ratio = prev.d > 0 ? Math.round((dropFromTop / prev.d) * 10) / 10 : 0;
    const minAmplitude = prev.d / 2;
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
  peak.topText = peak.juggleCount;
  peak.bottomText = String(result.ratio);
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

    pushBallState(smoothedX, smoothedY, dDisplay, false, t, vx, vy, null, null, null);
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
      pushBallState(predX, predY, d, true, t, undefined, undefined, null, null, null);
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
    const hasLabels = pt.topText != null || pt.bottomText != null;
    let labelTop = el.querySelector('.snake-dot-label-top');
    let labelBottom = el.querySelector('.snake-dot-label-bottom');
    if (hasLabels) {
      if (!labelTop) {
        labelTop = document.createElement('div');
        labelTop.setAttribute('class', 'snake-dot-label snake-dot-label-top');
        el.appendChild(labelTop);
      }
      if (!labelBottom) {
        labelBottom = document.createElement('div');
        labelBottom.setAttribute('class', 'snake-dot-label snake-dot-label-bottom');
        el.appendChild(labelBottom);
      }
      labelTop.textContent = pt.topText ?? '';
      labelBottom.textContent = pt.bottomText ?? '';
      labelTop.style.display = pt.topText != null ? 'block' : 'none';
      labelBottom.style.display = pt.bottomText != null ? 'block' : 'none';
    } else {
      if (labelTop) labelTop.style.display = 'none';
      if (labelBottom) labelBottom.style.display = 'none';
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
