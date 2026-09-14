let poseLandmarker;

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const { FilesetResolver, PoseLandmarker } = await import(`${data.visionUrl}/vision_bundle.mjs`);
      const vision = await FilesetResolver.forVisionTasks(`${data.visionUrl}/wasm`);
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: data.modelUrl, delegate: data.delegate },
        canvas: new OffscreenCanvas(1, 1),
        runningMode: 'VIDEO',
        numPoses: 1,
      });
      self.postMessage({ type: 'ready' });
    } else if (data.type === 'delegate') {
      await poseLandmarker.setOptions({ baseOptions: { delegate: data.delegate } });
      self.postMessage({ type: 'ready' });
    } else if (data.type === 'detect') {
      const start = performance.now();
      const result = poseLandmarker.detectForVideo(data.frame, data.timestamp);
      const poseMs = performance.now() - start;
      // Return only the small landmark arrays needed by the overlay.
      self.postMessage({ landmarks: result.landmarks, poseMs });
    }
  } catch (error) {
    self.postMessage({ error: error.message || String(error) });
  } finally {
    data.frame?.close();
  }
};
