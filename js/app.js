// Based on CodePen: https://codepen.io/mediapipe-preview/pen/vYrWvNg
// Guide: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/web_js
import { ObjectDetector, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/vision_bundle.mjs';
import { Kalman1D } from './kalman1d.js';

const demosSection = document.getElementById('demos');
const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const videoStage = document.getElementById('videoStage');
const aiMsEl = document.getElementById('aiMs');
const postAiMsEl = document.getElementById('postAiMs');
const totalMsFpsEl = document.getElementById('totalMsFps');
const juggleCountEl = document.getElementById('juggleCount');
const sessionCountEl = document.getElementById('sessionCount');
const sessionPrimaryBtn = document.getElementById('sessionPrimaryBtn');
const sessionStopBtn = document.getElementById('sessionStopBtn');
const sessionMenuBtn = document.getElementById('sessionMenuBtn');
const sessionRecEl = document.getElementById('sessionRec');
const sessionTimerEl = document.getElementById('sessionTimer');
const sessionHintEl = document.getElementById('sessionHint');
const settingsOverlay = document.getElementById('settingsOverlay');
const settingsCloseBtn = document.getElementById('settingsCloseBtn');
const settingsDoneBtn = document.getElementById('settingsDoneBtn');
const voiceCountCheckbox = document.getElementById('voiceCountCheckbox');
const showSnakeCheckbox = document.getElementById('showSnakeCheckbox');
const showBallCheckbox = document.getElementById('showBallCheckbox');
const showTimingCheckbox = document.getElementById('showTimingCheckbox');
const fileDebugCheckbox = document.getElementById('fileDebugCheckbox');
const fileDebugSettingRow = document.getElementById('fileDebugSettingRow');
const fromFileBtn = document.getElementById('fromFileBtn');
const liveSourceBtn = document.getElementById('liveSourceBtn');
const videoFileInput = document.getElementById('videoFileInput');
const fileTransportBar = document.getElementById('fileTransportBar');
const fileStepBackBtn = document.getElementById('fileStepBackBtn');
const fileStepForwardBtn = document.getElementById('fileStepForwardBtn');
const filePlayPauseBtn = document.getElementById('filePlayPauseBtn');
const timingStatsEl = document.getElementById('timingStats');

const FILE_FPS = 30;
const STATE_BUFFER_CAPACITY = Math.floor(window.innerWidth / 5);
const KALMAN_PROCESS_VARIANCE = 0.01;
const KALMAN_MEASUREMENT_VARIANCE = 0.1;
const AUTO_PAUSE_MS = 5000;
const AUTO_PAUSE_HINT_MS = 3000;
const SNAKE_DOT_SIZE = 5;
const SNAKE_DOT_SIZE_JUGGLE = 10;

const JUGGLE_COUNT_WORDS = [
  'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten',
  'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen', 'Twenty',
];

/** @type {{ session: 'notRunning'|'running'|'paused', videoSource: 'camera'|'file', fileObjectUrl: string|null, filePlaybackActive: boolean, fileStepTime: number, juggleCount: number, lastJugglePeakAt: number|null, timer: { startedAt: number|null, pausedAccumMs: number, pauseStartedAt: number|null }, ballState: object[], lastLocalMinY: number|null, kalman: { x: import('./kalman1d.js').Kalman1D|null, y: import('./kalman1d.js').Kalman1D|null, lastT: number|null }, settings: { voice: boolean, showSnake: boolean, showBall: boolean, showTiming: boolean, fileDebug: boolean }, lastVideoTime: number, autoPauseHintUntil: number }} */
const STATE = {
  session: 'notRunning',
  videoSource: 'camera',
  fileObjectUrl: null,
  filePlaybackActive: false,
  fileStepTime: 0,
  juggleCount: 0,
  lastJugglePeakAt: null,
  timer: {
    startedAt: null,
    pausedAccumMs: 0,
    pauseStartedAt: null,
  },
  ballState: [],
  lastLocalMinY: null,
  kalman: { x: null, y: null, lastT: null },
  settings: { voice: false, showSnake: true, showBall: true, showTiming: true, fileDebug: false },
  lastVideoTime: -1,
  autoPauseHintUntil: 0,
};

let objectDetector;
let runningMode = 'IMAGE';
let preferredVoice = null;
let rafId = null;
let ballHighlighter = null;
let snakeFrame = null;
let snakeDots = [];

function isTestHarnessPage() {
  return document.getElementById('testPanel') != null;
}

function isIndexPage() {
  return !isTestHarnessPage();
}

function isCameraSource() {
  return isIndexPage() && STATE.videoSource === 'camera';
}

function isVideoDisplayMirrored() {
  return isCameraSource();
}

function shouldRunDetection() {
  if (isTestHarnessPage()) return true;
  if (STATE.videoSource === 'file') return STATE.filePlaybackActive;
  return STATE.session === 'running';
}

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

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

function formatSessionTime(ms) {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return min + ':' + String(sec).padStart(2, '0');
}

function getSessionElapsedMs() {
  const { startedAt, pausedAccumMs, pauseStartedAt } = STATE.timer;
  if (startedAt == null) return 0;
  if (STATE.session === 'paused' && pauseStartedAt != null) {
    return pauseStartedAt - startedAt - pausedAccumMs;
  }
  return Date.now() - startedAt - pausedAccumMs;
}

function resetTrackingState() {
  STATE.ballState.length = 0;
  STATE.lastLocalMinY = null;
  STATE.kalman.x = null;
  STATE.kalman.y = null;
  STATE.kalman.lastT = null;
  hideTrackingVisuals();
}

function resetSessionTimer() {
  STATE.timer.startedAt = null;
  STATE.timer.pausedAccumMs = 0;
  STATE.timer.pauseStartedAt = null;
}

function isShowSnake() {
  if (!isIndexPage()) return true;
  return STATE.settings.showSnake;
}

function isShowBall() {
  if (!isIndexPage()) return true;
  return STATE.settings.showBall;
}

function isShowTiming() {
  if (!isIndexPage()) return true;
  return STATE.settings.showTiming;
}

function applyVisualizationSettings() {
  if (timingStatsEl) {
    timingStatsEl.classList.toggle('timing-stats--hidden', !isShowTiming());
  }
  if (!isShowBall() && ballHighlighter) ballHighlighter.style.display = 'none';
  if (!isShowSnake() && snakeFrame) snakeFrame.style.display = 'none';
}

function hideTrackingVisuals() {
  if (ballHighlighter) ballHighlighter.style.display = 'none';
  if (snakeFrame) snakeFrame.style.display = 'none';
}

function showAutoPauseHint() {
  if (!sessionHintEl) return;
  sessionHintEl.textContent = 'Paused — no juggles for 5s';
  sessionHintEl.classList.remove('hidden');
  STATE.autoPauseHintUntil = Date.now() + AUTO_PAUSE_HINT_MS;
}

function hideAutoPauseHint() {
  STATE.autoPauseHintUntil = 0;
  if (sessionHintEl) sessionHintEl.classList.add('hidden');
}

function setJuggleCount(n) {
  STATE.juggleCount = n;
  if (sessionCountEl) sessionCountEl.textContent = String(n);
  if (juggleCountEl) juggleCountEl.textContent = n + ' juggles';
  if (isVoiceEnabled()) speakJuggleCount(n);
}

function isVoiceEnabled() {
  if (isIndexPage()) return STATE.settings.voice;
  return voiceCountCheckbox?.checked ?? false;
}

function setPrimarySessionButton(mode) {
  if (!sessionPrimaryBtn) return;
  const playIcon = sessionPrimaryBtn.querySelector('.session-btn__icon--play');
  const pauseIcon = sessionPrimaryBtn.querySelector('.session-btn__icon--pause');
  sessionPrimaryBtn.classList.remove('session-btn--start', 'session-btn--pause', 'session-btn--resume');
  if (mode === 'pause') {
    sessionPrimaryBtn.classList.add('session-btn--pause');
    sessionPrimaryBtn.title = 'Pause';
    sessionPrimaryBtn.setAttribute('aria-label', 'Pause');
    if (playIcon) playIcon.hidden = true;
    if (pauseIcon) pauseIcon.hidden = false;
  } else if (mode === 'resume') {
    sessionPrimaryBtn.classList.add('session-btn--resume');
    sessionPrimaryBtn.title = 'Resume';
    sessionPrimaryBtn.setAttribute('aria-label', 'Resume');
    if (playIcon) playIcon.hidden = false;
    if (pauseIcon) pauseIcon.hidden = true;
  } else {
    sessionPrimaryBtn.classList.add('session-btn--start');
    sessionPrimaryBtn.title = 'Start';
    sessionPrimaryBtn.setAttribute('aria-label', 'Start');
    if (playIcon) playIcon.hidden = false;
    if (pauseIcon) pauseIcon.hidden = true;
  }
}

function formatVideoTime(sec) {
  const totalSec = Math.max(0, Math.floor(sec));
  const min = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return min + ':' + String(s).padStart(2, '0');
}

function updateFileTimeUI() {
  if (!isIndexPage() || STATE.videoSource !== 'file' || !sessionTimerEl) return;
  const dur = video.duration;
  if (!dur || !Number.isFinite(dur)) {
    sessionTimerEl.textContent = '0:00';
    return;
  }
  sessionTimerEl.textContent = formatVideoTime(video.currentTime) + ' / ' + formatVideoTime(dur);
}

function updateVideoSourceUI() {
  if (!isIndexPage()) return;
  document.body.classList.toggle('video-source-file', STATE.videoSource === 'file');
  document.body.classList.toggle('video-source-camera', STATE.videoSource === 'camera');
  if (liveSourceBtn) liveSourceBtn.disabled = STATE.videoSource === 'camera';
  const showTransport = STATE.videoSource === 'file' && STATE.settings.fileDebug;
  if (fileTransportBar) {
    fileTransportBar.classList.toggle('hidden', !showTransport);
    fileTransportBar.setAttribute('aria-hidden', showTransport ? 'false' : 'true');
  }
}

function updateSessionUI() {
  if (!isIndexPage() || STATE.videoSource === 'file') {
    if (sessionCountEl) sessionCountEl.textContent = String(STATE.juggleCount);
    if (STATE.videoSource === 'file') updateFileTimeUI();
    return;
  }

  if (sessionCountEl) sessionCountEl.textContent = String(STATE.juggleCount);

  if (STATE.autoPauseHintUntil > 0 && Date.now() > STATE.autoPauseHintUntil) {
    hideAutoPauseHint();
  }

  if (STATE.session === 'running') {
    setPrimarySessionButton('pause');
  } else if (STATE.session === 'paused') {
    setPrimarySessionButton('resume');
  } else {
    setPrimarySessionButton('start');
  }

  if (sessionStopBtn) {
    sessionStopBtn.disabled = STATE.session === 'notRunning';
  }

  if (sessionRecEl && sessionTimerEl) {
    sessionRecEl.classList.remove('session-rec--running', 'session-rec--paused', 'session-rec--idle');
    if (STATE.session === 'notRunning') {
      sessionRecEl.classList.add('session-rec--idle');
      sessionTimerEl.textContent = '0:00';
    } else if (STATE.session === 'running') {
      sessionRecEl.classList.add('session-rec--running');
      sessionTimerEl.textContent = formatSessionTime(getSessionElapsedMs());
    } else {
      sessionRecEl.classList.add('session-rec--paused');
      sessionTimerEl.textContent = formatSessionTime(getSessionElapsedMs());
    }
  }

  if (sessionCountEl) {
    sessionCountEl.classList.toggle('session-count--paused', STATE.session === 'paused');
  }
}

function startSession() {
  resetTrackingState();
  STATE.session = 'running';
  STATE.juggleCount = 0;
  STATE.lastJugglePeakAt = Date.now();
  resetSessionTimer();
  STATE.timer.startedAt = Date.now();
  hideAutoPauseHint();
  setJuggleCount(0);
  updateSessionUI();
}

function pauseSession(showHint) {
  if (STATE.session !== 'running') return;
  STATE.session = 'paused';
  if (STATE.timer.startedAt != null && STATE.timer.pauseStartedAt == null) {
    STATE.timer.pauseStartedAt = Date.now();
  }
  hideTrackingVisuals();
  if (showHint) showAutoPauseHint();
  updateSessionUI();
}

function resumeSession() {
  if (STATE.session !== 'paused') return;
  if (STATE.timer.pauseStartedAt != null) {
    STATE.timer.pausedAccumMs += Date.now() - STATE.timer.pauseStartedAt;
    STATE.timer.pauseStartedAt = null;
  }
  STATE.session = 'running';
  STATE.lastJugglePeakAt = Date.now();
  resetTrackingState();
  hideAutoPauseHint();
  updateSessionUI();
}

function stopSession() {
  STATE.session = 'notRunning';
  STATE.juggleCount = 0;
  STATE.lastJugglePeakAt = null;
  resetSessionTimer();
  resetTrackingState();
  hideAutoPauseHint();
  setJuggleCount(0);
  updateSessionUI();
}

function checkAutoPause() {
  if (STATE.session !== 'running' || STATE.lastJugglePeakAt == null) return;
  if (Date.now() - STATE.lastJugglePeakAt >= AUTO_PAUSE_MS) {
    pauseSession(true);
  }
}

function openSettings() {
  if (!settingsOverlay) return;
  if (voiceCountCheckbox) voiceCountCheckbox.checked = STATE.settings.voice;
  if (showSnakeCheckbox) showSnakeCheckbox.checked = STATE.settings.showSnake;
  if (showBallCheckbox) showBallCheckbox.checked = STATE.settings.showBall;
  if (showTimingCheckbox) showTimingCheckbox.checked = STATE.settings.showTiming;
  if (fileDebugCheckbox) fileDebugCheckbox.checked = STATE.settings.fileDebug;
  updateVideoSourceUI();
  settingsOverlay.classList.remove('hidden');
  settingsOverlay.setAttribute('aria-hidden', 'false');
}

function closeSettings() {
  if (!settingsOverlay) return;
  settingsOverlay.classList.add('hidden');
  settingsOverlay.setAttribute('aria-hidden', 'true');
}

function syncSettingsFromUI() {
  if (voiceCountCheckbox) STATE.settings.voice = voiceCountCheckbox.checked;
  if (showSnakeCheckbox) STATE.settings.showSnake = showSnakeCheckbox.checked;
  if (showBallCheckbox) STATE.settings.showBall = showBallCheckbox.checked;
  if (showTimingCheckbox) STATE.settings.showTiming = showTimingCheckbox.checked;
  if (fileDebugCheckbox) STATE.settings.fileDebug = fileDebugCheckbox.checked;
  applyVisualizationSettings();
  updateVideoSourceUI();
}

function stopCameraStream() {
  const stream = video.srcObject;
  if (stream && typeof stream.getTracks === 'function') {
    stream.getTracks().forEach((track) => track.stop());
  }
  video.srcObject = null;
}

function stopFrameLoop() {
  if (rafId != null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  video.pause();
  STATE.filePlaybackActive = false;
}

function releaseFileObjectUrl() {
  if (STATE.fileObjectUrl) {
    URL.revokeObjectURL(STATE.fileObjectUrl);
    STATE.fileObjectUrl = null;
  }
}

function beginFileCountingSession() {
  resetTrackingState();
  STATE.juggleCount = 0;
  setJuggleCount(0);
  STATE.filePlaybackActive = true;
  STATE.lastVideoTime = -1;
}

function updateFilePlayPauseLabel(playing) {
  if (!filePlayPauseBtn) return;
  filePlayPauseBtn.textContent = playing ? 'Pause' : 'Play';
  filePlayPauseBtn.title = playing ? 'Pause' : 'Play';
  filePlayPauseBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
}

function runFileRealtimeLoop(onEnded) {
  stopFrameLoop();
  STATE.filePlaybackActive = true;
  let lastProcessedFrame = -1;
  function step() {
    if (video.ended || (video.duration && video.currentTime >= video.duration - 0.001)) {
      stopFrameLoop();
      updateFileTimeUI();
      if (onEnded) onEnded(STATE.juggleCount);
      return;
    }
    const currentFrame = Math.floor(video.currentTime * FILE_FPS);
    if (currentFrame > lastProcessedFrame) {
      lastProcessedFrame = currentFrame;
      runOneDetectionFrame().then(() => {
        updateSessionUI();
        rafId = requestAnimationFrame(step);
      });
    } else {
      rafId = requestAnimationFrame(step);
    }
  }
  updateFilePlayPauseLabel(true);
  video.play().then(() => {
    rafId = requestAnimationFrame(step);
  }).catch((err) => {
    console.error(err);
    updateFilePlayPauseLabel(false);
  });
}

function seekAndDetectFileFrame(targetTime, onDone) {
  const dur = video.duration || 0;
  const t = Math.max(0, Math.min(targetTime, Math.max(0, dur - 0.001)));
  STATE.fileStepTime = t;
  const runDetect = () => {
    runOneDetectionFrame().then(() => {
      updateSessionUI();
      if (onDone) onDone();
    });
  };
  if (Math.abs(video.currentTime - t) < 0.0001 && video.readyState >= 2) {
    runDetect();
    return;
  }
  video.currentTime = t;
  video.addEventListener('seeked', function onSeeked() {
    video.removeEventListener('seeked', onSeeked);
    runDetect();
  }, { once: true });
}

function fileStepBack() {
  stopFrameLoop();
  updateFilePlayPauseLabel(false);
  const t = (STATE.fileStepTime || video.currentTime) - 1 / FILE_FPS;
  seekAndDetectFileFrame(t);
}

function fileStepForward() {
  stopFrameLoop();
  updateFilePlayPauseLabel(false);
  const t = (STATE.fileStepTime || video.currentTime) + 1 / FILE_FPS;
  if (video.duration && t >= video.duration) return;
  seekAndDetectFileFrame(t);
}

function filePlayPauseToggle() {
  if (rafId != null) {
    stopFrameLoop();
    updateFilePlayPauseLabel(false);
    return;
  }
  runFileRealtimeLoop();
}

function onFileVideoLoaded(debug, onComplete) {
  resizeStageToContain();
  window.addEventListener('resize', resizeStageToContain);
  beginFileCountingSession();
  updateVideoSourceUI();
  if (debug) {
    STATE.fileStepTime = 0;
    updateFilePlayPauseLabel(false);
    seekAndDetectFileFrame(0, onComplete);
  } else {
    runFileRealtimeLoop(onComplete);
  }
}

function loadFileVideo(url, options = {}) {
  const { debug = false, onEnded } = options;
  stopFrameLoop();
  stopCameraStream();
  releaseFileObjectUrl();
  if (url.startsWith('blob:')) STATE.fileObjectUrl = url;
  video.src = url;
  video.load();
  return new Promise((resolve, reject) => {
    video.addEventListener('loadeddata', function onLoaded() {
      video.removeEventListener('loadeddata', onLoaded);
      onFileVideoLoaded(debug, () => {
        const count = STATE.juggleCount;
        if (onEnded) onEnded(count);
        resolve(count);
      });
    }, { once: true });
    video.addEventListener('error', function onError() {
      video.removeEventListener('error', onError);
      reject(new Error('Video failed to load'));
    }, { once: true });
  });
}

async function switchToFile(file) {
  if (!file || !objectDetector) return;
  stopSession();
  STATE.videoSource = 'file';
  document.body.classList.add('live-active');
  liveView?.classList.add('live-fullscreen');
  const url = URL.createObjectURL(file);
  try {
    await loadFileVideo(url, { debug: STATE.settings.fileDebug });
  } catch (err) {
    console.error(err);
    releaseFileObjectUrl();
    STATE.videoSource = 'camera';
    updateVideoSourceUI();
  }
}

async function switchToLive() {
  stopSession();
  stopFrameLoop();
  releaseFileObjectUrl();
  video.removeAttribute('src');
  video.load();
  STATE.videoSource = 'camera';
  STATE.fileStepTime = 0;
  updateVideoSourceUI();
  updateSessionUI();
  if (objectDetector && hasGetUserMedia()) {
    await enableCam();
  }
}

function initSessionUI() {
  if (!isIndexPage()) return;

  sessionPrimaryBtn?.addEventListener('click', () => {
    if (STATE.session === 'notRunning') startSession();
    else if (STATE.session === 'running') pauseSession(false);
    else if (STATE.session === 'paused') resumeSession();
  });

  sessionStopBtn?.addEventListener('click', () => {
    if (STATE.session !== 'notRunning') stopSession();
  });

  sessionMenuBtn?.addEventListener('click', openSettings);
  settingsCloseBtn?.addEventListener('click', closeSettings);
  settingsDoneBtn?.addEventListener('click', closeSettings);
  settingsOverlay?.addEventListener('click', (e) => {
    if (e.target === settingsOverlay) closeSettings();
  });
  voiceCountCheckbox?.addEventListener('change', syncSettingsFromUI);
  showSnakeCheckbox?.addEventListener('change', syncSettingsFromUI);
  showBallCheckbox?.addEventListener('change', syncSettingsFromUI);
  showTimingCheckbox?.addEventListener('change', syncSettingsFromUI);
  fileDebugCheckbox?.addEventListener('change', syncSettingsFromUI);

  fromFileBtn?.addEventListener('click', () => videoFileInput?.click());
  videoFileInput?.addEventListener('change', () => {
    const file = videoFileInput.files?.[0];
    if (file) {
      syncSettingsFromUI();
      closeSettings();
      switchToFile(file);
    }
    videoFileInput.value = '';
  });
  liveSourceBtn?.addEventListener('click', () => {
    if (STATE.videoSource !== 'camera') {
      closeSettings();
      switchToLive();
    }
  });
  fileStepBackBtn?.addEventListener('click', fileStepBack);
  fileStepForwardBtn?.addEventListener('click', fileStepForward);
  filePlayPauseBtn?.addEventListener('click', filePlayPauseToggle);

  applyVisualizationSettings();
  updateVideoSourceUI();
  updateSessionUI();
}

initSessionUI();

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
  if (isIndexPage() && isCameraSource() && hasGetUserMedia()) {
    enableCam();
  } else if (isIndexPage() && isCameraSource()) {
    console.warn('getUserMedia() is not supported by your browser');
  }
};
initializeObjectDetector();

if (hasGetUserMedia() && isIndexPage()) {
  document.body.classList.add('live-active');
  liveView.classList.add('live-fullscreen');
}

async function enableCam() {
  if (!objectDetector || STATE.videoSource !== 'camera') return;

  stopFrameLoop();
  releaseFileObjectUrl();
  video.removeAttribute('src');
  video.load();

  const constraints = { video: { facingMode: 'user' } };

  navigator.mediaDevices
    .getUserMedia(constraints)
    .then(function (stream) {
      stopCameraStream();
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

/**
 * Run one frame of detection (used by frame-driven video test only).
 */
async function runOneDetectionFrame() {
  if (runningMode === 'IMAGE') {
    runningMode = 'VIDEO';
    await objectDetector.setOptions({ runningMode: 'VIDEO' });
  }
  const startTimeMs = performance.now();
  STATE.lastVideoTime = video.currentTime;
  const detections = objectDetector.detectForVideo(video, startTimeMs);
  displayVideoDetections(detections);
}

async function predictWebcam() {
  if (STATE.videoSource !== 'camera' || isTestHarnessPage()) {
    rafId = null;
    return;
  }
  if (video.ended) {
    rafId = null;
    return;
  }

  updateSessionUI();
  checkAutoPause();

  const t0 = performance.now();
  let hadNewFrame = false;
  let detectForVideoMs = 0;

  if (shouldRunDetection()) {
    if (runningMode === 'IMAGE') {
      runningMode = 'VIDEO';
      await objectDetector.setOptions({ runningMode: 'VIDEO' });
    }
    const startTimeMs = performance.now();

    if (video.currentTime !== STATE.lastVideoTime) {
      STATE.lastVideoTime = video.currentTime;
      const t1 = performance.now();
      const detections = objectDetector.detectForVideo(video, startTimeMs);
      const t2 = performance.now();
      detectForVideoMs = Math.round(t2 - t1);
      hadNewFrame = true;
      displayVideoDetections(detections);
    }
  } else {
    hideTrackingVisuals();
  }

  const t3 = performance.now();
  const predictWebcamMs = Math.round(t3 - t0);
  if (hadNewFrame && isShowTiming() && aiMsEl && postAiMsEl && totalMsFpsEl) {
    const postAiMs = predictWebcamMs - detectForVideoMs;
    const fps = predictWebcamMs > 0 ? 1000 / predictWebcamMs : 0;
    aiMsEl.textContent = 'AI: ' + detectForVideoMs + ' ms';
    postAiMsEl.textContent = 'PostAI: ' + postAiMs + ' ms';
    totalMsFpsEl.textContent = 'Total: ' + predictWebcamMs + ' ms / ' + fps.toFixed(1) + ' FPS';
  }
  rafId = window.requestAnimationFrame(predictWebcam);
}

function pushBallState(x, y, d, calculatedOnly, t, vx, vy, juggleCount = null, topText = null, bottomText = null) {
  let vxOut = vx != null ? vx : 0;
  let vyOut = vy != null ? vy : 0;
  if (STATE.ballState.length > 0 && vxOut === 0 && vyOut === 0) {
    const prev = STATE.ballState[STATE.ballState.length - 1];
    const dtSec = (t - prev.t) / 1000;
    if (dtSec > 0) {
      vxOut = (x - prev.x) / dtSec;
      vyOut = (y - prev.y) / dtSec;
    }
  }
  STATE.ballState.push({ x, y, vx: vxOut, vy: vyOut, d, calculatedOnly, t, juggleCount: juggleCount ?? null, topText: topText ?? null, bottomText: bottomText ?? null });
  if (STATE.ballState.length > STATE_BUFFER_CAPACITY) STATE.ballState.shift();
}

function isNewJuggleDetected() {
  const detected = STATE.ballState.filter((e) => !e.calculatedOnly);
  if (detected.length < 3) return { isJuggleDetected: false, ratio: null };
  const n = detected.length;
  const prev = detected[n - 2];
  const curr = detected[n - 1];
  const prevPrev = detected[n - 3];
  if (prev.y <= prevPrev.y && prev.y <= curr.y) {
    STATE.lastLocalMinY = prev.y;
  }
  if (prev.y >= prevPrev.y && prev.y >= curr.y) {
    const dropFromTop = prev.y - (STATE.lastLocalMinY != null ? STATE.lastLocalMinY : prev.y);
    const ratio = prev.d > 0 ? Math.round((dropFromTop / prev.d) * 10) / 10 : 0;
    const minAmplitude = prev.d / 2;
    const isJuggleDetected = dropFromTop >= minAmplitude;
    return { isJuggleDetected, ratio };
  }
  return { isJuggleDetected: false, ratio: null };
}

function setJuggleInBallState(result) {
  if (result.ratio == null || STATE.ballState.length < 2) return;
  const peak = STATE.ballState[STATE.ballState.length - 2];
  if (result.isJuggleDetected) {
    setJuggleCount(STATE.juggleCount + 1);
    STATE.lastJugglePeakAt = Date.now();
    peak.juggleCount = STATE.juggleCount;
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
  const dtSec = STATE.kalman.lastT != null ? (t - STATE.kalman.lastT) / 1000 : 0;
  STATE.kalman.lastT = t;

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

    if (!STATE.kalman.x) {
      STATE.kalman.x = new Kalman1D(KALMAN_PROCESS_VARIANCE, KALMAN_MEASUREMENT_VARIANCE);
      STATE.kalman.y = new Kalman1D(KALMAN_PROCESS_VARIANCE, KALMAN_MEASUREMENT_VARIANCE);
    }
    if (!STATE.kalman.x.initialised) {
      STATE.kalman.x.x[0] = centerXDisplay;
      STATE.kalman.x.x[1] = 0;
      STATE.kalman.x.x[2] = 0;
      STATE.kalman.x.initialised = true;
    }
    if (!STATE.kalman.y.initialised) {
      STATE.kalman.y.x[0] = centerYDisplay;
      STATE.kalman.y.x[1] = 0;
      STATE.kalman.y.x[2] = 0;
      STATE.kalman.y.initialised = true;
    }
    STATE.kalman.x.update(centerXDisplay);
    STATE.kalman.y.update(centerYDisplay);
    const smoothedX = STATE.kalman.x.x[0];
    const smoothedY = STATE.kalman.y.x[0];
    const vx = STATE.kalman.x.x[1];
    const vy = STATE.kalman.y.x[1];
    STATE.kalman.x.predict(dtSec);
    STATE.kalman.y.predict(dtSec);

    pushBallState(smoothedX, smoothedY, dDisplay, false, t, vx, vy, null, null, null);
    const juggleResult = isNewJuggleDetected();
    if (juggleResult.ratio != null) setJuggleInBallState(juggleResult);

    if (isShowBall()) {
      const ballLeft = isVideoDisplayMirrored()
        ? dw - centerXDisplay - dDisplay / 2
        : centerXDisplay - dDisplay / 2;
      ballHighlighter.style.left = ballLeft + 'px';
      ballHighlighter.style.top = (centerYDisplay - dDisplay / 2) + 'px';
      ballHighlighter.style.width = dDisplay + 'px';
      ballHighlighter.style.height = dDisplay + 'px';
      ballHighlighter.style.display = 'block';
    } else {
      ballHighlighter.style.display = 'none';
    }
  } else {
    ballHighlighter.style.display = 'none';
    if (STATE.kalman.x && STATE.kalman.y && STATE.kalman.x.initialised) {
      const predX = STATE.kalman.x.predict(dtSec);
      const predY = STATE.kalman.y.predict(dtSec);
      const d = STATE.ballState.length > 0 ? STATE.ballState[STATE.ballState.length - 1].d : 40;
      pushBallState(predX, predY, d, true, t, undefined, undefined, null, null, null);
    }
  }
  liveSnakeVisualisation();
}

function liveSnakeVisualisation() {
  if (!isShowSnake()) {
    if (snakeFrame) snakeFrame.style.display = 'none';
    return;
  }
  const n = STATE.ballState.length;
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

  let minY = STATE.ballState[0].y;
  let maxY = STATE.ballState[0].y;
  for (let i = 1; i < n; i++) {
    const y = STATE.ballState[i].y;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const rangeY = maxY - minY;
  const yScale = rangeY > 0 ? 1 / rangeY : 0;

  for (let i = 0; i < n; i++) {
    const pt = STATE.ballState[i];
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
  stopSession();
  STATE.lastVideoTime = -1;
  if (rafId != null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (juggleCountEl) juggleCountEl.textContent = '0 juggles';
}

function runTestHarnessAutoDebug(onEnded) {
  let nextTime = 0;
  function step() {
    if (nextTime >= video.duration) {
      stopFrameLoop();
      if (onEnded) onEnded(STATE.juggleCount);
      return;
    }
    video.currentTime = nextTime;
    video.addEventListener('seeked', function onSeeked() {
      video.removeEventListener('seeked', onSeeked);
      runOneDetectionFrame().then(() => {
        nextTime += 1 / FILE_FPS;
        rafId = requestAnimationFrame(step);
      });
    }, { once: true });
  }
  step();
}

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
    window.runJuggleTestStart = function start(debugMode) {
      window.runJuggleTestStart = null;
      beginFileCountingSession();
      if (debugMode) {
        runTestHarnessAutoDebug(resolveResult);
      } else {
        runFileRealtimeLoop(resolveResult);
      }
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
