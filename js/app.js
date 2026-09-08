// Based on CodePen: https://codepen.io/mediapipe-preview/pen/vYrWvNg
// Guide: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/web_js
import { ObjectDetector, PoseLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs';
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
const helpOpenBtn = document.getElementById('helpOpenBtn');
const helpOverlay = document.getElementById('helpOverlay');
const helpCloseBtn = document.getElementById('helpCloseBtn');
const helpOkBtn = document.getElementById('helpOkBtn');
const shareOpenBtn = document.getElementById('shareOpenBtn');
const shareOverlay = document.getElementById('shareOverlay');
const shareCloseBtn = document.getElementById('shareCloseBtn');
const shareOkBtn = document.getElementById('shareOkBtn');
const shareLinkBtn = document.getElementById('shareLinkBtn');
const shareCopyBtn = document.getElementById('shareCopyBtn');
const shareNativeBtn = document.getElementById('shareNativeBtn');
const shareCopyStatus = document.getElementById('shareCopyStatus');
const voiceVolumeSlider = document.getElementById('voiceVolumeSlider');
const voiceVolumeValueEl = document.getElementById('voiceVolumeValue');
const voiceEverySlider = document.getElementById('voiceEverySlider');
const voiceEveryValueEl = document.getElementById('voiceEveryValue');
const handsFreeCheckbox = document.getElementById('handsFreeCheckbox');
const autoPauseCheckbox = document.getElementById('autoPauseCheckbox');
const minBounceSlider = document.getElementById('minBounceSlider');
const minBounceValueEl = document.getElementById('minBounceValue');
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
const fileScrubber = document.getElementById('fileScrubber');
const fileScrubFrameEl = document.getElementById('fileScrubFrame');
const timingStatsEl = document.getElementById('timingStats');
const pwaInstallBtn = document.getElementById('pwaInstallBtn');

let fileScrubberSyncing = false;
let fileScrubThrottleId = null;
let deferredPwaInstallPrompt = null;
let pwaInstallDismissedThisLoad = false;

const FILE_FPS = 30;
const STATE_BUFFER_CAPACITY = Math.floor(window.innerWidth / 5);
const KALMAN_PROCESS_VARIANCE = 0.01;
const KALMAN_MEASUREMENT_VARIANCE = 0.1;
const AUTO_PAUSE_MS = 5000;
const AUTO_PAUSE_FILL_DELAY_MS = 2000;
const AUTO_PAUSE_FILL_MS = AUTO_PAUSE_MS - AUTO_PAUSE_FILL_DELAY_MS;
const AUTO_PAUSE_HINT_MS = 3000;
const SNAKE_DOT_SIZE = 5;
const SNAKE_DOT_SIZE_JUGGLE = 10;
const VOICE_EVERY_N_OPTIONS = [1, 5, 10, 25, 50];
const POSE_HOLD_MS = 1000;
const POSE_COOLDOWN_MS = 500;
const POSE_VISIBILITY_MIN = 0.5;
const POSE_LANDMARKER_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const APP_SHARE_URL = 'https://andruwik777.github.io/JuggleJs/';
const APP_SHARE_TITLE = 'Football Juggling';

const JUGGLE_COUNT_WORDS = [
  'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten',
  'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen', 'Twenty',
];

/** @type {{ session: 'notRunning'|'running'|'paused', videoSource: 'camera'|'file', fileObjectUrl: string|null, filePlaybackActive: boolean, fileStepTime: number, juggleCount: number, lastJugglePeakAt: number|null, timer: { startedAt: number|null, pausedAccumMs: number, pauseStartedAt: number|null }, ballState: object[], lastLocalMinY: number|null, kalman: { x: import('./kalman1d.js').Kalman1D|null, y: import('./kalman1d.js').Kalman1D|null, lastT: number|null }, settings: { voiceVolume: number, voiceEveryN: number, handsFree: boolean, autoPause: boolean, minBounce: number, showSnake: boolean, showBall: boolean, showTiming: boolean, fileDebug: boolean }, lastVideoTime: number, autoPauseHintUntil: number, pose: { holdAction: null|'start'|'stop', holdSince: number|null, ignoreUntil: number, needsNeutral: boolean } }} */
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
  settings: {
    voiceVolume: 0.5,
    voiceEveryN: 1,
    handsFree: true,
    autoPause: true,
    minBounce: 0.2,
    showSnake: true,
    showBall: true,
    showTiming: true,
    fileDebug: true,
  },
  lastVideoTime: -1,
  autoPauseHintUntil: 0,
  pose: {
    holdAction: null,
    holdSince: null,
    ignoreUntil: 0,
    needsNeutral: false,
  },
};

let objectDetector;
let poseLandmarker;
let runningMode = 'IMAGE';
let poseRunningMode = 'IMAGE';
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
  if (n < 1) return;
  const every = STATE.settings.voiceEveryN > 0 ? STATE.settings.voiceEveryN : 1;
  if (n % every !== 0) return;
  speakVoiceWord(JUGGLE_COUNT_WORDS[n - 1] ?? String(n));
}

function speakVoiceWord(word) {
  if (!isVoiceEnabled() || typeof speechSynthesis === 'undefined') return;
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(word);
  utterance.lang = 'en-US';
  utterance.rate = 1.1;
  utterance.volume = getVoiceVolume();
  if (preferredVoice) utterance.voice = preferredVoice;
  speechSynthesis.speak(utterance);
}

initVoiceCount();

function isHandsFreeEnabled() {
  return isIndexPage() && STATE.settings.handsFree;
}

function shouldRunPoseControls() {
  return (
    isHandsFreeEnabled() &&
    isCameraSource() &&
    poseLandmarker != null &&
    STATE.session !== 'running' &&
    !!video.srcObject
  );
}

function landmarkVisible(lm) {
  return lm && (lm.visibility == null || lm.visibility >= POSE_VISIBILITY_MIN);
}

function isArmsUpPose(landmarks) {
  const ls = landmarks[11];
  const rs = landmarks[12];
  const lw = landmarks[15];
  const rw = landmarks[16];
  if (![ls, rs, lw, rw].every(landmarkVisible)) return false;
  const margin = 0.02;
  return lw.y < ls.y - margin && rw.y < rs.y - margin;
}

function isArmsCrossedPose(landmarks) {
  const ls = landmarks[11];
  const rs = landmarks[12];
  const lw = landmarks[15];
  const rw = landmarks[16];
  const lh = landmarks[23];
  const rh = landmarks[24];
  if (![ls, rs, lw, rw, lh, rh].every(landmarkVisible)) return false;

  const shoulderY = (ls.y + rs.y) / 2;
  const hipY = (lh.y + rh.y) / 2;
  const torso = Math.max(0.15, Math.abs(hipY - shoulderY));
  const inChest = (w) => w.y > shoulderY - 0.08 * torso && w.y < shoulderY + 0.55 * torso;
  if (!inChest(lw) || !inChest(rw)) return false;

  const midX = (ls.x + rs.x) / 2;
  const crossed = (lw.x - midX) * (rw.x - midX) < 0;
  const shoulderWidth = Math.max(0.08, Math.abs(ls.x - rs.x));
  const wristsClose = Math.abs(lw.x - rw.x) < shoulderWidth * 1.8;
  const closeY = Math.abs(lw.y - rw.y) < 0.25 * torso + 0.06;
  return crossed && wristsClose && closeY;
}

function setPoseButtonProgress(btn, progress) {
  if (!btn) return;
  if (progress <= 0) {
    btn.classList.remove('session-btn--pose-filling');
    btn.style.removeProperty('--pose-progress');
    return;
  }
  btn.classList.add('session-btn--pose-filling');
  btn.style.setProperty('--pose-progress', String(Math.min(1, Math.max(0, progress))));
}

function clearPoseUiProgress() {
  setPoseButtonProgress(sessionPrimaryBtn, 0);
  setPoseButtonProgress(sessionStopBtn, 0);
}

function resetPoseHoldState(options = {}) {
  STATE.pose.holdAction = null;
  STATE.pose.holdSince = null;
  if (options.clearUi !== false) clearPoseUiProgress();
}

function beginPoseCooldown() {
  STATE.pose.ignoreUntil = Date.now() + POSE_COOLDOWN_MS;
  STATE.pose.needsNeutral = true;
  resetPoseHoldState();
}

function classifyPoseAction(landmarks) {
  const armsUp = isArmsUpPose(landmarks);
  const armsCrossed = isArmsCrossedPose(landmarks);
  if (armsUp && armsCrossed) return null;
  if (armsUp) return 'start';
  if (armsCrossed) return 'stop';
  return null;
}

function applyPoseHoldProgress(action, now) {
  if (STATE.pose.holdAction !== action) {
    STATE.pose.holdAction = action;
    STATE.pose.holdSince = now;
  }
  const since = STATE.pose.holdSince ?? now;
  const progress = Math.min(1, (now - since) / POSE_HOLD_MS);
  if (action === 'start') {
    setPoseButtonProgress(sessionPrimaryBtn, progress);
    setPoseButtonProgress(sessionStopBtn, 0);
  } else {
    setPoseButtonProgress(sessionStopBtn, progress);
    setPoseButtonProgress(sessionPrimaryBtn, 0);
  }
  return progress >= 1;
}

async function processHandsFreePoseFrame() {
  if (!shouldRunPoseControls()) {
    resetPoseHoldState();
    return 0;
  }

  const now = Date.now();
  if (now < STATE.pose.ignoreUntil) {
    resetPoseHoldState();
    return 0;
  }

  if (poseRunningMode === 'IMAGE') {
    poseRunningMode = 'VIDEO';
    await poseLandmarker.setOptions({ runningMode: 'VIDEO' });
  }

  const t1 = performance.now();
  const result = poseLandmarker.detectForVideo(video, t1);
  const detectMs = Math.round(performance.now() - t1);
  const landmarks = result?.landmarks?.[0];
  if (!landmarks) {
    STATE.pose.needsNeutral = false;
    resetPoseHoldState();
    return detectMs;
  }

  const action = classifyPoseAction(landmarks);
  if (!action) {
    STATE.pose.needsNeutral = false;
    resetPoseHoldState();
    return detectMs;
  }

  if (STATE.pose.needsNeutral) {
    resetPoseHoldState();
    return detectMs;
  }

  if (action === 'start' && (STATE.session === 'notRunning' || STATE.session === 'paused')) {
    if (applyPoseHoldProgress('start', now)) {
      beginPoseCooldown();
      if (STATE.session === 'notRunning') startSession();
      else resumeSession();
    }
  } else if (action === 'stop' && STATE.session === 'paused') {
    if (applyPoseHoldProgress('stop', now)) {
      beginPoseCooldown();
      stopSession();
    }
  } else {
    resetPoseHoldState();
  }

  return detectMs;
}

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
  const prev = STATE.juggleCount;
  STATE.juggleCount = n;
  if (sessionCountEl) sessionCountEl.textContent = String(n);
  if (juggleCountEl) juggleCountEl.textContent = n + ' juggles';
  if (isVoiceEnabled() && n !== prev) speakJuggleCount(n);
}

function isVoiceEnabled() {
  if (!isIndexPage()) {
    const cb = document.getElementById('voiceCountCheckbox');
    return cb?.checked ?? STATE.settings.voiceVolume > 0;
  }
  return STATE.settings.voiceVolume > 0;
}

function getVoiceVolume() {
  if (!isIndexPage()) {
    return isVoiceEnabled() ? 1 : 0;
  }
  return Math.max(0, Math.min(1, STATE.settings.voiceVolume));
}

function formatVoiceVolumeLabel(volume) {
  if (volume <= 0) return 'OFF';
  if (volume >= 1) return 'MAX';
  return Math.round(volume * 100) + '%';
}

function syncVoiceVolumeLabel() {
  if (voiceVolumeValueEl) {
    voiceVolumeValueEl.textContent = formatVoiceVolumeLabel(STATE.settings.voiceVolume);
  }
}

function voiceEveryNToSliderIndex(everyN) {
  const idx = VOICE_EVERY_N_OPTIONS.indexOf(everyN);
  return idx >= 0 ? idx : 0;
}

function syncVoiceEveryLabel() {
  if (voiceEveryValueEl) {
    voiceEveryValueEl.textContent = String(STATE.settings.voiceEveryN);
  }
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
  if (!isIndexPage() || STATE.videoSource !== 'file') return;
  const dur = video.duration;
  if (sessionTimerEl) {
    if (!dur || !Number.isFinite(dur)) {
      sessionTimerEl.textContent = '0:00';
    } else {
      sessionTimerEl.textContent = formatVideoTime(video.currentTime) + ' / ' + formatVideoTime(dur);
    }
  }
  syncFileScrubberFromVideo();
  updateFileScrubFrameLabel();
}

function syncFileScrubberFromVideo() {
  if (!fileScrubber || fileScrubberSyncing) return;
  const dur = video.duration;
  if (!dur || !Number.isFinite(dur)) return;
  fileScrubberSyncing = true;
  fileScrubber.max = String(dur);
  fileScrubber.value = String(video.currentTime);
  fileScrubberSyncing = false;
}

function updateFileScrubFrameLabel() {
  if (!fileScrubFrameEl) return;
  const dur = video.duration;
  if (!dur || !Number.isFinite(dur)) {
    fileScrubFrameEl.textContent = '';
    return;
  }
  const frame = Math.floor(video.currentTime * FILE_FPS);
  const total = Math.max(0, Math.floor(dur * FILE_FPS));
  fileScrubFrameEl.textContent = 'Frame ' + frame + ' / ' + total;
}

function getCurrentFileVideoFrame() {
  return Math.floor(video.currentTime * FILE_FPS);
}

function fileVideoFrameToTime(frame) {
  return frame / FILE_FPS;
}

function recalculateJuggleCountFromBallState() {
  let max = 0;
  for (const pt of STATE.ballState) {
    if (pt.juggleCount != null && pt.juggleCount > max) max = pt.juggleCount;
  }
  setJuggleCount(max);
}

function recomputeLastLocalMinYFromBallState() {
  const detected = STATE.ballState.filter((e) => !e.calculatedOnly);
  STATE.lastLocalMinY = null;
  if (detected.length < 3) return;
  for (let i = 2; i < detected.length; i++) {
    const prevPrev = detected[i - 2];
    const prev = detected[i - 1];
    const curr = detected[i];
    if (prev.y <= prevPrev.y && prev.y <= curr.y) {
      STATE.lastLocalMinY = prev.y;
    }
  }
}

function trimBallStateBeforeFileFrame(targetFrame) {
  STATE.ballState = STATE.ballState.filter(
    (pt) => pt.fileVideoFrame == null || pt.fileVideoFrame < targetFrame
  );
}

function scrubFileToTime(targetTime) {
  stopFrameLoop();
  updateFilePlayPauseLabel(false);
  if (fileScrubThrottleId != null) {
    clearTimeout(fileScrubThrottleId);
    fileScrubThrottleId = null;
  }
  resetTrackingState();
  setJuggleCount(0);
  STATE.filePlaybackActive = true;
  seekAndDetectFileFrame(targetTime);
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

function startSession(options = {}) {
  const announce = options.announce !== false;
  resetTrackingState();
  STATE.session = 'running';
  STATE.juggleCount = 0;
  STATE.lastJugglePeakAt = Date.now();
  resetSessionTimer();
  STATE.timer.startedAt = Date.now();
  hideAutoPauseHint();
  clearPoseUiProgress();
  if (announce) speakVoiceWord('Started');
  setJuggleCount(0);
  updateSessionUI();
}

function pauseSession(showHint, options = {}) {
  if (STATE.session !== 'running') return;
  const announce = options.announce !== false;
  STATE.session = 'paused';
  if (STATE.timer.startedAt != null && STATE.timer.pauseStartedAt == null) {
    STATE.timer.pauseStartedAt = Date.now();
  }
  hideTrackingVisuals();
  clearPoseUiProgress();
  if (showHint) showAutoPauseHint();
  if (announce) speakVoiceWord('Paused');
  updateSessionUI();
}

function resumeSession(options = {}) {
  if (STATE.session !== 'paused') return;
  const announce = options.announce !== false;
  if (STATE.timer.pauseStartedAt != null) {
    STATE.timer.pausedAccumMs += Date.now() - STATE.timer.pauseStartedAt;
    STATE.timer.pauseStartedAt = null;
  }
  STATE.session = 'running';
  STATE.lastJugglePeakAt = Date.now();
  resetTrackingState();
  hideAutoPauseHint();
  clearPoseUiProgress();
  if (announce) speakVoiceWord('Resumed');
  updateSessionUI();
}

function stopSession(options = {}) {
  const announce = options.announce !== false && STATE.session !== 'notRunning';
  STATE.session = 'notRunning';
  STATE.juggleCount = 0;
  STATE.lastJugglePeakAt = null;
  resetSessionTimer();
  resetTrackingState();
  hideAutoPauseHint();
  clearPoseUiProgress();
  if (announce) speakVoiceWord('Stopped');
  setJuggleCount(0);
  updateSessionUI();
}

function updateAutoPauseProgressUi() {
  if (
    !STATE.settings.autoPause ||
    STATE.session !== 'running' ||
    STATE.lastJugglePeakAt == null ||
    !sessionPrimaryBtn?.classList.contains('session-btn--pause')
  ) {
    if (sessionPrimaryBtn?.classList.contains('session-btn--pause')) {
      setPoseButtonProgress(sessionPrimaryBtn, 0);
    }
    return;
  }
  const idleMs = Date.now() - STATE.lastJugglePeakAt;
  if (idleMs < AUTO_PAUSE_FILL_DELAY_MS) {
    setPoseButtonProgress(sessionPrimaryBtn, 0);
    return;
  }
  const progress = Math.min(1, (idleMs - AUTO_PAUSE_FILL_DELAY_MS) / AUTO_PAUSE_FILL_MS);
  setPoseButtonProgress(sessionPrimaryBtn, progress);
}

function checkAutoPause() {
  if (!STATE.settings.autoPause) {
    if (sessionPrimaryBtn?.classList.contains('session-btn--pause')) {
      setPoseButtonProgress(sessionPrimaryBtn, 0);
    }
    return;
  }
  if (STATE.session !== 'running' || STATE.lastJugglePeakAt == null) return;
  updateAutoPauseProgressUi();
  if (Date.now() - STATE.lastJugglePeakAt >= AUTO_PAUSE_MS) {
    pauseSession(true);
  }
}

function openSettings() {
  if (!settingsOverlay) return;
  if (voiceVolumeSlider) {
    voiceVolumeSlider.value = String(STATE.settings.voiceVolume);
    syncVoiceVolumeLabel();
  }
  if (voiceEverySlider) {
    voiceEverySlider.value = String(voiceEveryNToSliderIndex(STATE.settings.voiceEveryN));
    syncVoiceEveryLabel();
  }
  if (handsFreeCheckbox) handsFreeCheckbox.checked = STATE.settings.handsFree;
  if (autoPauseCheckbox) autoPauseCheckbox.checked = STATE.settings.autoPause;
  if (minBounceSlider) {
    minBounceSlider.value = String(STATE.settings.minBounce);
    if (minBounceValueEl) minBounceValueEl.textContent = String(STATE.settings.minBounce);
  }
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

function openHelp() {
  if (!helpOverlay) return;
  closeSettings();
  helpOverlay.classList.remove('hidden');
  helpOverlay.setAttribute('aria-hidden', 'false');
  const body = helpOverlay.querySelector('.help-body');
  if (body) body.scrollTop = 0;
}

function closeHelp() {
  if (!helpOverlay) return;
  helpOverlay.classList.add('hidden');
  helpOverlay.setAttribute('aria-hidden', 'true');
}

function openShare() {
  if (!shareOverlay) return;
  closeSettings();
  if (shareCopyStatus) shareCopyStatus.textContent = '';
  if (shareNativeBtn) {
    const canShare = typeof navigator.share === 'function';
    shareNativeBtn.classList.toggle('hidden', !canShare);
  }
  shareOverlay.classList.remove('hidden');
  shareOverlay.setAttribute('aria-hidden', 'false');
  const body = shareOverlay.querySelector('.help-body');
  if (body) body.scrollTop = 0;
}

function closeShare() {
  if (!shareOverlay) return;
  shareOverlay.classList.add('hidden');
  shareOverlay.setAttribute('aria-hidden', 'true');
}

async function copyShareUrl() {
  if (shareCopyStatus) shareCopyStatus.textContent = '';
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      await navigator.clipboard.writeText(APP_SHARE_URL);
    } else {
      const ta = document.createElement('textarea');
      ta.value = APP_SHARE_URL;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    if (shareCopyStatus) shareCopyStatus.textContent = 'Copied';
  } catch (err) {
    console.warn('Copy failed', err);
    if (shareCopyStatus) shareCopyStatus.textContent = 'Copy failed';
  }
}

async function nativeShareApp() {
  if (typeof navigator.share !== 'function') return;
  try {
    await navigator.share({
      title: APP_SHARE_TITLE,
      text: 'Count football juggles with your phone camera.',
      url: APP_SHARE_URL,
    });
  } catch (err) {
    if (err && err.name === 'AbortError') return;
    console.warn('Share failed', err);
  }
}

function syncSettingsFromUI() {
  if (voiceVolumeSlider) {
    const v = parseFloat(voiceVolumeSlider.value);
    if (Number.isFinite(v)) {
      STATE.settings.voiceVolume = Math.round(v * 10) / 10;
      syncVoiceVolumeLabel();
    }
  }
  if (voiceEverySlider) {
    const idx = parseInt(voiceEverySlider.value, 10);
    if (Number.isFinite(idx) && VOICE_EVERY_N_OPTIONS[idx] != null) {
      STATE.settings.voiceEveryN = VOICE_EVERY_N_OPTIONS[idx];
      syncVoiceEveryLabel();
    }
  }
  if (handsFreeCheckbox) STATE.settings.handsFree = handsFreeCheckbox.checked;
  if (autoPauseCheckbox) STATE.settings.autoPause = autoPauseCheckbox.checked;
  if (minBounceSlider) {
    const v = parseFloat(minBounceSlider.value);
    if (Number.isFinite(v)) {
      STATE.settings.minBounce = Math.round(v * 10) / 10;
      if (minBounceValueEl) minBounceValueEl.textContent = String(STATE.settings.minBounce);
    }
  }
  if (showSnakeCheckbox) STATE.settings.showSnake = showSnakeCheckbox.checked;
  if (showBallCheckbox) STATE.settings.showBall = showBallCheckbox.checked;
  if (showTimingCheckbox) STATE.settings.showTiming = showTimingCheckbox.checked;
  if (fileDebugCheckbox) STATE.settings.fileDebug = fileDebugCheckbox.checked;
  if (!STATE.settings.handsFree) resetPoseHoldState();
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
  updateFilePlayPauseLabel(false);
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
  const currentFrame = getCurrentFileVideoFrame();
  if (currentFrame <= 0) return;
  const targetFrame = currentFrame - 1;
  trimBallStateBeforeFileFrame(targetFrame);
  recalculateJuggleCountFromBallState();
  recomputeLastLocalMinYFromBallState();
  STATE.kalman.x = null;
  STATE.kalman.y = null;
  STATE.kalman.lastT = null;
  liveSnakeVisualisation();
  STATE.filePlaybackActive = true;
  seekAndDetectFileFrame(fileVideoFrameToTime(targetFrame));
}

function fileStepForward() {
  stopFrameLoop();
  updateFilePlayPauseLabel(false);
  const t = video.currentTime + 1 / FILE_FPS;
  if (video.duration && t >= video.duration) return;
  seekAndDetectFileFrame(t);
}

function isFileVideoAtEnd() {
  const dur = video.duration;
  if (!dur || !Number.isFinite(dur)) return false;
  return video.ended || video.currentTime >= dur - 0.001;
}

function filePlayPauseToggle() {
  if (rafId != null) {
    stopFrameLoop();
    updateFilePlayPauseLabel(false);
    return;
  }
  if (isFileVideoAtEnd()) {
    resetTrackingState();
    setJuggleCount(0);
    STATE.lastVideoTime = -1;
    STATE.fileStepTime = 0;
    const startFromBeginning = () => runFileRealtimeLoop();
    if (video.currentTime <= 0.001) {
      startFromBeginning();
      return;
    }
    video.addEventListener('seeked', function onSeeked() {
      video.removeEventListener('seeked', onSeeked);
      startFromBeginning();
    }, { once: true });
    video.currentTime = 0;
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
    // File mode must not free-run: index.html video has autoplay for the camera.
    video.pause();
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
  video.removeAttribute('autoplay');
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
  stopSession({ announce: false });
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
  stopSession({ announce: false });
  stopFrameLoop();
  releaseFileObjectUrl();
  video.removeAttribute('src');
  video.setAttribute('autoplay', '');
  video.setAttribute('playsinline', '');
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
  helpOpenBtn?.addEventListener('click', openHelp);
  helpCloseBtn?.addEventListener('click', closeHelp);
  helpOkBtn?.addEventListener('click', closeHelp);
  shareOpenBtn?.addEventListener('click', openShare);
  shareCloseBtn?.addEventListener('click', closeShare);
  shareOkBtn?.addEventListener('click', closeShare);
  shareCopyBtn?.addEventListener('click', copyShareUrl);
  shareLinkBtn?.addEventListener('click', copyShareUrl);
  shareNativeBtn?.addEventListener('click', nativeShareApp);
  voiceVolumeSlider?.addEventListener('input', syncSettingsFromUI);
  voiceEverySlider?.addEventListener('input', syncSettingsFromUI);
  handsFreeCheckbox?.addEventListener('change', syncSettingsFromUI);
  autoPauseCheckbox?.addEventListener('change', syncSettingsFromUI);
  minBounceSlider?.addEventListener('input', syncSettingsFromUI);
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
  fileScrubber?.addEventListener('pointerdown', () => {
    stopFrameLoop();
    updateFilePlayPauseLabel(false);
  });
  fileScrubber?.addEventListener('input', () => {
    if (fileScrubberSyncing || !fileScrubber) return;
    const t = parseFloat(fileScrubber.value);
    if (!Number.isFinite(t)) return;
    if (fileScrubThrottleId != null) clearTimeout(fileScrubThrottleId);
    fileScrubThrottleId = setTimeout(() => scrubFileToTime(t), 80);
  });
  fileScrubber?.addEventListener('change', () => {
    if (fileScrubberSyncing || !fileScrubber) return;
    if (fileScrubThrottleId != null) {
      clearTimeout(fileScrubThrottleId);
      fileScrubThrottleId = null;
    }
    scrubFileToTime(parseFloat(fileScrubber.value));
  });

  applyVisualizationSettings();
  updateVideoSourceUI();
  updateSessionUI();
  initPwaInstallUi();
}

function isRunningAsInstalledPwa() {
  return (
    window.matchMedia('(display-mode: standalone)').matches ||
    window.matchMedia('(display-mode: fullscreen)').matches ||
    window.navigator.standalone === true
  );
}

function hidePwaInstallButton() {
  if (pwaInstallBtn) pwaInstallBtn.classList.add('hidden');
}

function showPwaInstallButton() {
  if (!pwaInstallBtn || !isIndexPage()) return;
  if (pwaInstallDismissedThisLoad || isRunningAsInstalledPwa() || !deferredPwaInstallPrompt) return;
  pwaInstallBtn.classList.remove('hidden');
}

function dismissPwaInstallPromptUi() {
  pwaInstallDismissedThisLoad = true;
  hidePwaInstallButton();
}

async function onPwaInstallButtonClick() {
  if (!deferredPwaInstallPrompt) {
    hidePwaInstallButton();
    return;
  }
  const promptEvent = deferredPwaInstallPrompt;
  deferredPwaInstallPrompt = null;
  hidePwaInstallButton();
  try {
    await promptEvent.prompt();
    await promptEvent.userChoice;
  } catch (err) {
    console.warn('PWA install prompt failed', err);
  }
}

function initPwaInstallUi() {
  if (!isIndexPage() || !pwaInstallBtn) return;

  if (isRunningAsInstalledPwa()) {
    hidePwaInstallButton();
    return;
  }

  window.addEventListener('beforeinstallprompt', (event) => {
    event.preventDefault();
    deferredPwaInstallPrompt = event;
    showPwaInstallButton();
  });

  window.addEventListener('appinstalled', () => {
    deferredPwaInstallPrompt = null;
    hidePwaInstallButton();
  });

  pwaInstallBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    onPwaInstallButtonClick();
  });

  document.addEventListener('click', (event) => {
    if (pwaInstallBtn.classList.contains('hidden')) return;
    const target = event.target;
    if (!(target instanceof Element)) return;
    const btn = target.closest('button');
    if (!btn || btn === pwaInstallBtn) return;
    dismissPwaInstallPromptUi();
  }, true);
}

function registerServiceWorker() {
  if (!('serviceWorker' in navigator) || !isIndexPage()) return;
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch((err) => {
      console.warn('Service worker registration failed', err);
    });
  });
}

initSessionUI();
registerServiceWorker();

const initializeVisionTasks = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm'
  );
  const MODEL_PATH = './models/model_fp16.tflite';
  const DETECTION_CATEGORY_NAME = 'Juggling - v7 2022-07-26 4-53pm';

  const objectDetectorPromise = ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_PATH,
      delegate: 'GPU'
    },
    scoreThreshold: 0.4,
    maxResults: 1,
    runningMode: runningMode,
    categoryAllowlist: [DETECTION_CATEGORY_NAME]
  });

  const poseLandmarkerPromise = isIndexPage()
    ? PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: POSE_LANDMARKER_MODEL,
          delegate: 'GPU'
        },
        runningMode: poseRunningMode,
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      }).catch((err) => {
        console.warn('Pose Landmarker failed to load; hands-free disabled', err);
        return null;
      })
    : Promise.resolve(null);

  objectDetector = await objectDetectorPromise;
  poseLandmarker = await poseLandmarkerPromise;

  demosSection.classList.remove('invisible');
  window.dispatchEvent(new Event('juggleAppReady'));
  if (isIndexPage() && isCameraSource() && hasGetUserMedia()) {
    enableCam();
  } else if (isIndexPage() && isCameraSource()) {
    console.warn('getUserMedia() is not supported by your browser');
  }
};
initializeVisionTasks();

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

  const t0 = performance.now();
  let hadNewFrame = false;
  let detectForVideoMs = 0;

  if (shouldRunDetection()) {
    // Keep autopause fill on the Pause button; only reset pose-hold bookkeeping.
    resetPoseHoldState({ clearUi: false });
    setPoseButtonProgress(sessionStopBtn, 0);
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
    if (video.currentTime !== STATE.lastVideoTime) {
      STATE.lastVideoTime = video.currentTime;
      detectForVideoMs = await processHandsFreePoseFrame();
      hadNewFrame = detectForVideoMs > 0 || shouldRunPoseControls();
    } else if (!shouldRunPoseControls()) {
      resetPoseHoldState();
    }
  }

  checkAutoPause();

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
  const entry = {
    x, y, vx: vxOut, vy: vyOut, d, calculatedOnly, t,
    juggleCount: juggleCount ?? null,
    topText: topText ?? null,
    bottomText: bottomText ?? null,
  };
  if (isIndexPage() && STATE.videoSource === 'file') {
    entry.fileVideoFrame = getCurrentFileVideoFrame();
  }
  STATE.ballState.push(entry);
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
    const minAmplitude = prev.d * STATE.settings.minBounce;
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
  let sumD = 0;
  for (let i = 0; i < n; i++) {
    const pt = STATE.ballState[i];
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
    if (pt.d > 0) sumD += pt.d;
  }
  const dRef = sumD > 0 ? sumD / n : 40;
  const rawRangeY = maxY - minY;
  const minRangeY = SNAKE_MIN_RANGE_BALL_FRACTION * dRef;
  const rangeY = Math.max(rawRangeY, minRangeY);
  const yScale = 1 / rangeY;
  const snakeFloorMode = rawRangeY < minRangeY;

  for (let i = 0; i < n; i++) {
    const pt = STATE.ballState[i];
    const dotSize = pt.juggleCount != null ? SNAKE_DOT_SIZE_JUGGLE : SNAKE_DOT_SIZE;
    const half = dotSize / 2;
    const cap = STATE_BUFFER_CAPACITY;
    const xFrac = cap > 1 ? ((cap - n) + i) / (cap - 1) : 0.5;
    const x = xFrac * frameW;
    const yFrac = snakeFloorMode ? 1 : (pt.y - minY) * yScale;
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
  stopSession({ announce: false });
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
