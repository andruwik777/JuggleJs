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
const detectForVideoMsEl = document.getElementById('detectForVideoMs');
const predictWebcamMsEl = document.getElementById('predictWebcamMs');
const deltaMsEl = document.getElementById('deltaMs');
let enableWebcamButton;

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

var children = [];

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
      video.addEventListener('loadeddata', predictWebcam);
    })
    .catch((err) => {
      console.error(err);
    });
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

function displayVideoDetections(result) {
  for (let child of children) {
    liveView.removeChild(child);
  }
  children.splice(0);

  for (let detection of result.detections) {
    const p = document.createElement('p');
    p.innerText =
      detection.categories[0].categoryName +
      ' - with ' +
      Math.round(parseFloat(detection.categories[0].score) * 100) +
      '% confidence.';
    p.style =
      'left: ' +
      (video.offsetWidth -
        detection.boundingBox.width -
        detection.boundingBox.originX) +
      'px;' +
      'top: ' +
      detection.boundingBox.originY +
      'px; ' +
      'width: ' +
      (detection.boundingBox.width - 10) +
      'px;';

    const highlighter = document.createElement('div');
    highlighter.setAttribute('class', 'highlighter');
    highlighter.style =
      'left: ' +
      (video.offsetWidth -
        detection.boundingBox.width -
        detection.boundingBox.originX) +
      'px;' +
      'top: ' +
      detection.boundingBox.originY +
      'px;' +
      'width: ' +
      (detection.boundingBox.width - 10) +
      'px;' +
      'height: ' +
      detection.boundingBox.height +
      'px;';

    liveView.appendChild(highlighter);
    liveView.appendChild(p);

    children.push(highlighter);
    children.push(p);
  }
}
