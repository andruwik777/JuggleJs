// MediaPipe Tasks Vision (ES module per https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/web_js)
import { FilesetResolver, ObjectDetector } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';

(function (FilesetResolverRef, ObjectDetectorRef) {
  'use strict';

  // --- Constants (plan: category name and model path) ---
  var DETECTION_CATEGORY_NAME = 'Juggling - v7 2022-07-26 4-53pm';
  var MODEL_PATH = './models/model_fp16.tflite';
  var POSITION_BUFFER_SIZE = 8;
  var MIN_JUGGLE_INTERVAL_MS = 400;
  var WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';

  // --- DOM refs ---
  var video = document.getElementById('video');
  var canvasOverlay = document.getElementById('canvas-overlay');
  var ctx = canvasOverlay.getContext('2d');
  var countDisplay = document.getElementById('count-display');
  var inferenceMsEl = document.getElementById('inference-ms');
  var errorMsg = document.getElementById('error-msg');
  var btnStart = document.getElementById('btn-start');
  var btnStop = document.getElementById('btn-stop');
  var btnReset = document.getElementById('btn-reset');

  // --- State ---
  var stream = null;
  var objectDetector = null;
  var animationId = null;
  var isRunning = false;
  var juggleCount = 0;
  var positionBuffer = [];
  var lastJuggleTime = 0;
  var lastVideoTime = -1;
  var lastDetectionTime = 0;
  var DETECTION_INTERVAL_MS = 33; // ~30 FPS for live camera (guide: process when frame changes)

  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.add('visible');
  }

  function clearError() {
    errorMsg.textContent = '';
    errorMsg.classList.remove('visible');
  }

  function setCount(value) {
    juggleCount = value;
    countDisplay.textContent = String(juggleCount);
  }

  function resetCounterState() {
    positionBuffer = [];
    lastJuggleTime = 0;
    setCount(0);
  }

  // --- Camera: single video element as frame source (E2E-ready) ---
  function startCamera() {
    if (stream) return Promise.resolve();
    clearError();
    var constraints = {
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    };
    return navigator.mediaDevices.getUserMedia(constraints)
      .then(function (s) {
        stream = s;
        video.srcObject = stream;
        return new Promise(function (resolve) {
          video.onloadedmetadata = function () {
            video.play().then(resolve).catch(resolve);
          };
        });
      })
      .catch(function (err) {
        showError('Camera error: ' + (err.message || 'Permission denied'));
        throw err;
      });
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
      video.srcObject = null;
    }
  }

  // --- MediaPipe Object Detector ---
  function initDetector() {
    if (objectDetector) return Promise.resolve(objectDetector);
    return FilesetResolverRef.forVisionTasks(WASM_PATH).then(function (vision) {
      return ObjectDetectorRef.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH },
        scoreThreshold: 0.4,
        maxResults: 5,
        runningMode: 'VIDEO',
        categoryAllowlist: [DETECTION_CATEGORY_NAME]
      });
    }).then(function (detector) {
      objectDetector = detector;
      return objectDetector;
    }).catch(function (err) {
      showError('Model load error: ' + (err.message || String(err)));
      throw err;
    });
  }

  function getBallCenterFromDetections(detections) {
    if (!detections || !detections.detections || detections.detections.length === 0) return null;
    var best = null;
    for (var i = 0; i < detections.detections.length; i++) {
      var d = detections.detections[i];
      var cat = d.categories && d.categories[0];
      if (!cat || cat.categoryName !== DETECTION_CATEGORY_NAME) continue;
      var bbox = d.boundingBox;
      if (!bbox) continue;
      var cx = bbox.originX + bbox.width / 2;
      var cy = bbox.originY + bbox.height / 2;
      if (!best || (cat.score && cat.score > (best.score || 0))) {
        best = { x: cx, y: cy, score: cat.score, boundingBox: bbox };
      }
    }
    return best;
  }

  // --- Juggle count: local max y (ball at lowest point) ---
  function pushPosition(center) {
    if (!center) return;
    positionBuffer.push({ x: center.x, y: center.y, t: Date.now() });
    if (positionBuffer.length > POSITION_BUFFER_SIZE) positionBuffer.shift();
  }

  function tryCountJuggle() {
    if (positionBuffer.length < 3) return;
    var n = positionBuffer.length;
    var prev = positionBuffer[n - 2];
    var curr = positionBuffer[n - 1];
    var prevPrev = positionBuffer[n - 3];
    if (prev.y >= prevPrev.y && prev.y >= curr.y) {
      var now = Date.now();
      if (now - lastJuggleTime >= MIN_JUGGLE_INTERVAL_MS) {
        lastJuggleTime = now;
        setCount(juggleCount + 1);
      }
    }
  }

  // --- Bbox drawing (optional) ---
  function resizeCanvas() {
    var container = video.parentElement;
    var w = container.clientWidth;
    var h = container.clientHeight;
    if (canvasOverlay.width !== w || canvasOverlay.height !== h) {
      canvasOverlay.width = w;
      canvasOverlay.height = h;
    }
  }

  function drawBbox(center) {
    resizeCanvas();
    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
    if (!center || !center.boundingBox) return;
    var bbox = center.boundingBox;
    var vw = video.videoWidth;
    var vh = video.videoHeight;
    if (!vw || !vh) return;
    var scaleX = canvasOverlay.width / vw;
    var scaleY = canvasOverlay.height / vh;
    var x = bbox.originX * scaleX;
    var y = bbox.originY * scaleY;
    var w = bbox.width * scaleX;
    var h = bbox.height * scaleY;
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
  }

  // --- Detection loop: aligned with guide; for live camera use fixed interval (video.currentTime often does not advance per frame) ---
  function isLiveStream() {
    return video.srcObject && !video.src;
  }

  function runDetectionLoop() {
    if (!isRunning || !objectDetector || video.readyState < 2) {
      animationId = requestAnimationFrame(runDetectionLoop);
      return;
    }
    var now = performance.now();
    var shouldDetect = false;
    var timestampMs;
    if (isLiveStream()) {
      if (now - lastDetectionTime >= DETECTION_INTERVAL_MS) {
        shouldDetect = true;
        timestampMs = Math.round(now);
        lastDetectionTime = now;
      }
    } else {
      var currentTime = video.currentTime;
      if (currentTime !== lastVideoTime && currentTime >= 0) {
        shouldDetect = true;
        lastVideoTime = currentTime;
        timestampMs = Math.round(currentTime * 1000);
      }
    }
    if (shouldDetect) {
      var t0 = performance.now();
      var detections = objectDetector.detectForVideo(video, timestampMs);
      var inferenceMs = Math.round(performance.now() - t0);
      inferenceMsEl.textContent = inferenceMs + ' ms';
      var center = getBallCenterFromDetections(detections);
      pushPosition(center);
      tryCountJuggle();
      drawBbox(center);
    }
    animationId = requestAnimationFrame(runDetectionLoop);
  }

  function startLoop() {
    if (isRunning) return;
    isRunning = true;
    lastVideoTime = -1;
    lastDetectionTime = 0;
    inferenceMsEl.textContent = '';
    runDetectionLoop();
  }

  function stopLoop() {
    isRunning = false;
    if (animationId != null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
    inferenceMsEl.textContent = '';
    ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
  }

  // --- UI ---
  function onStart() {
    clearError();
    startCamera()
      .then(initDetector)
      .then(function () {
        startLoop();
        btnStart.disabled = true;
        btnStop.disabled = false;
      })
      .catch(function () {});
  }

  function onStop() {
    stopLoop();
    btnStart.disabled = false;
    btnStop.disabled = true;
  }

  function onReset() {
    resetCounterState();
  }

  btnStart.addEventListener('click', onStart);
  btnStop.addEventListener('click', onStop);
  btnReset.addEventListener('click', onReset);

  // Resize canvas when window resizes
  window.addEventListener('resize', resizeCanvas);
})(FilesetResolver, ObjectDetector);
