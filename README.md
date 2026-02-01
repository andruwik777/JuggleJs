# Juggle Counter (PoC)

Web app that counts football juggles using the device camera and MediaPipe Object Detection with a custom TFLite model.

## Setup

1. Add your model file: place `model_fp16.tflite` in the [models/](models/) folder (see [models/README.md](models/README.md)).
2. Serve the project over HTTP (required for camera and model loading). From the project root:

   ```bash
   npx serve .
   ```
   Or use WebStorm’s built-in static server; ensure the document root is the project root so `./models/model_fp16.tflite` and `./js/app.js` resolve correctly.

3. Open `http://localhost:3000` (or the port shown). Use `https` or `localhost` so the camera and APIs work.

## Usage

- **Start** — requests camera permission (if needed), loads the model, and starts detection and juggle counting.
- **Stop** — stops the detection loop; the camera can keep showing the picture.
- **Reset** — sets the count back to 0 and clears the internal position buffer.

The counter increases when the ball’s vertical position reaches a local maximum (lowest point in frame, i.e. touch). A green rectangle is drawn around the detected ball when running.

## E2E (future)

The same logic runs on a single `<video>` element. For E2E, use a video file as the source (e.g. `video.src = 'tests/fixtures/juggle_10.mp4'`) and run the same detection/count loop until the video ends, then assert the final count (e.g. 10).
