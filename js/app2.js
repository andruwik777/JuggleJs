// Ball detection runs on the main thread; pose detection runs in a worker.
const VISION_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
const POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const context = canvas.getContext('2d');
const status = document.getElementById('status');
const retry = document.getElementById('retry');
const timing = document.getElementById('timing');
const fpsValue = document.getElementById('fpsValue');
const poseGpu = document.getElementById('poseGpu');
const ballGpu = document.getElementById('ballGpu');
const ballBitmap = document.getElementById('ballBitmap');
const ballEnabled = document.getElementById('ballEnabled');
const poseEnabled = document.getElementById('poseEnabled');
let settingsRevision = 0;
let poseDelegate = 'CPU';
let ballDelegate = 'GPU';

let objectDetector;
let poseWorker;
let pendingPose;
let runId = 0;
let drawing;
let poseConnections;
let stream;
let animationId;
let lastVideoTime = -1;
let previousSample = null;
let timingTotals = {};
let timingCount = 0;

function resetTiming() {
  previousSample = null;
  timingTotals = {};
  timingCount = 0;
  fpsValue.textContent = '—';
  timing.textContent = 'Collecting timing samples…';
}

function recordInterval(nextStart) {
  if (!previousSample) return;
  const sample = previousSample;
  sample.Work = sample.end - sample.start;
  sample.Idle = nextStart - sample.end;
  sample.Interval = nextStart - sample.start;
  for (const name of ['Ball', 'Pose', 'Sum', 'Join', 'Prep', 'Wait', 'Frame', 'Cleanup', 'Tail', 'Work', 'Idle', 'Interval']) {
    timingTotals[name] = (timingTotals[name] || 0) + sample[name];
  }
  timingCount++;
  // Use the same complete start-to-start intervals for every average and FPS.
  if (timingTotals.Interval >= 500) {
    timing.textContent = Object.entries(timingTotals).map(([name, total]) => {
      const disabled = (name === 'Ball' && !sample.detectBall) || (name === 'Pose' && !sample.detectPose);
      return name + ': ' + (disabled ? 'off' : (total / timingCount).toFixed(1));
    }).join('\n');
    fpsValue.textContent = sample.detectBall || sample.detectPose
      ? (timingCount * 1000 / timingTotals.Interval).toFixed(1) : '0.0';
    timingTotals = {};
    timingCount = 0;
  }
}

function stop() {
  ballGpu.disabled = true;
  poseGpu.disabled = true;
  runId++;
  poseWorker?.terminate();
  poseWorker = null;
  pendingPose?.reject(new Error('Detection stopped.'));
  pendingPose = null;
  cancelAnimationFrame(animationId);
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function showError(error) {
  stop();
  console.error(error);
  status.textContent = `Unable to run the demo: ${error.message}`;
  retry.hidden = false;
}

function workerRequest(message, transfer = []) {
  return new Promise((resolve, reject) => {
    pendingPose = { resolve, reject };
    poseWorker.postMessage(message, transfer);
  });
}

async function initializePoseWorker() {
  // A classic worker also supports the WASM loader's importScripts calls.
  poseWorker = new Worker(new URL('./pose-worker2.js', import.meta.url));
  poseWorker.onmessage = ({ data }) => {
    const request = pendingPose;
    pendingPose = null;
    if (data.error) request?.reject(new Error(data.error));
    else request?.resolve(data);
  };
  poseWorker.onerror = (event) => {
    const error = new Error(event.message || 'Pose worker failed.');
    if (pendingPose) {
      pendingPose.reject(error);
      pendingPose = null;
    } else showError(error);
  };
  poseDelegate = poseGpu.checked ? 'GPU' : 'CPU';
  await workerRequest({ type: 'init', visionUrl: VISION_URL, modelUrl: POSE_MODEL_URL, delegate: poseDelegate });
}

async function renderFrame(currentRun = runId) {
  let frame;
  let workerFrame;
  let completedSample;
  try {
    // Apply changes between frames, when no detection request is pending.
    const requestedBallDelegate = ballGpu.checked ? 'GPU' : 'CPU';
    if (ballEnabled.checked && requestedBallDelegate !== ballDelegate) {
      ballGpu.disabled = true;
      status.textContent = `Switching ball to ${requestedBallDelegate}…`;
      await objectDetector.setOptions({ baseOptions: { delegate: requestedBallDelegate } });
      if (currentRun !== runId) return;
      ballDelegate = requestedBallDelegate;
      resetTiming();
      status.textContent = '';
      ballGpu.disabled = false;
    }
    const requestedDelegate = poseGpu.checked ? 'GPU' : 'CPU';
    if (poseEnabled.checked && requestedDelegate !== poseDelegate) {
      poseGpu.disabled = true;
      status.textContent = `Switching pose to ${requestedDelegate}…`;
      await workerRequest({ type: 'delegate', delegate: requestedDelegate });
      if (currentRun !== runId) return;
      poseDelegate = requestedDelegate;
      resetTiming();
      status.textContent = '';
      poseGpu.disabled = false;
    }
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
      const frameStart = performance.now();
      recordInterval(frameStart);
      previousSample = null;
      const revision = settingsRevision;
      const detectBall = ballEnabled.checked;
      const detectPose = poseEnabled.checked;
      const useBallBitmap = detectBall && ballBitmap.checked;
      lastVideoTime = video.currentTime;
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // Share a frozen image only in bitmap mode; video mode reads the live video.
      if (detectPose || useBallBitmap) {
        frame = await createImageBitmap(video);
        if (detectPose) workerFrame = useBallBitmap ? await createImageBitmap(frame) : frame;
      }
      if (currentRun !== runId) return;
      if (revision !== settingsRevision) {
        animationId = requestAnimationFrame(() => renderFrame(currentRun));
        return;
      }
      const timestamp = performance.now();
      const posePromise = detectPose
        ? workerRequest({ type: 'detect', frame: workerFrame, timestamp }, [workerFrame])
        : Promise.resolve({ landmarks: [], poseMs: 0 });
      // Handle cancellation even if object detection throws before the await.
      posePromise.catch(() => {});
      const objectStart = performance.now();
      const objects = detectBall ? objectDetector.detectForVideo(useBallBitmap ? frame : video, timestamp) : { detections: [] };
      const objectEnd = performance.now();
      const poses = await posePromise;
      const joinedAt = performance.now();
      if (currentRun !== runId) return;
      if (revision !== settingsRevision) {
        animationId = requestAnimationFrame(() => renderFrame(currentRun));
        return;
      }

      context.clearRect(0, 0, canvas.width, canvas.height);
      for (const landmarks of poses.landmarks) {
        drawing.drawConnectors(landmarks, poseConnections, { color: '#00ff80', lineWidth: 3 });
        drawing.drawLandmarks(landmarks, { color: '#ffffff', radius: 3 });
      }
      context.strokeStyle = '#ffdf00';
      context.lineWidth = 3;
      for (const detection of objects.detections) {
        const box = detection.boundingBox;
        if (box) context.strokeRect(box.originX, box.originY, box.width, box.height);
      }
      const frameEnd = performance.now();
      const ballMs = detectBall ? objectEnd - objectStart : 0;
      completedSample = {
        start: frameStart, frameEnd, revision, detectBall, detectPose,
        Ball: ballMs, Pose: poses.poseMs, Sum: ballMs + poses.poseMs,
        Join: joinedAt - timestamp, Prep: timestamp - frameStart,
        Wait: joinedAt - objectEnd, Frame: frameEnd - frameStart,
      };
    }
    // Only one frame is in flight: slow inference cannot build a stale frame queue.
    if (currentRun === runId) animationId = requestAnimationFrame(() => renderFrame(currentRun));
  } catch (error) {
    if (currentRun === runId) showError(error);
  } finally {
    const cleanupStart = performance.now();
    frame?.close();
    if (workerFrame !== frame) workerFrame?.close();
    const cleanupEnd = performance.now();
    if (completedSample && currentRun === runId && completedSample.revision === settingsRevision) {
      completedSample.Cleanup = cleanupEnd - cleanupStart;
      completedSample.Tail = cleanupEnd - completedSample.frameEnd;
      completedSample.end = cleanupEnd;
      previousSample = completedSample;
    }
  }
}

async function start() {
  ballGpu.disabled = true;
  poseGpu.disabled = true;
  retry.hidden = true;
  status.textContent = 'Loading models and starting the camera…';
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera access requires HTTPS or localhost and a supported browser.');
    }
    const { FilesetResolver, ObjectDetector, PoseLandmarker, DrawingUtils } = await import(`${VISION_URL}/vision_bundle.mjs`);
    const vision = await FilesetResolver.forVisionTasks(`${VISION_URL}/wasm`);
    if (!objectDetector) ballDelegate = ballGpu.checked ? 'GPU' : 'CPU';
    objectDetector ??= await ObjectDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: './models/model_fp16.tflite', delegate: ballDelegate },
      runningMode: 'VIDEO',
      scoreThreshold: 0.4,
      maxResults: 1,
      categoryAllowlist: ['Juggling - v7 2022-07-26 4-53pm'],
    });
    await initializePoseWorker();
    drawing = new DrawingUtils(context);
    poseConnections = PoseLandmarker.POSE_CONNECTIONS;

    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    lastVideoTime = -1;
    resetTiming();
    status.textContent = '';
    poseGpu.disabled = false;
    ballGpu.disabled = false;
    renderFrame();
  } catch (error) {
    showError(error);
  }
}

retry.addEventListener('click', start);
for (const checkbox of [ballEnabled, poseEnabled, ballBitmap]) {
  checkbox.addEventListener('change', () => {
    settingsRevision++;
    context.clearRect(0, 0, canvas.width, canvas.height);
    resetTiming();
  });
}
window.addEventListener('pagehide', () => {
  stop();
  objectDetector?.close();
  objectDetector = null;
});
window.addEventListener('pageshow', (event) => {
  if (event.persisted) start();
});
start();
