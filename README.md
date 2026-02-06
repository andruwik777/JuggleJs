# Juggle Counter (PoC)

Web app that counts **football juggles** using the device camera and MediaPipe Object Detection with a custom TFLite model.

Inspired by https://github.com/Logan1904/JuggleNet.git

## Try in your browser

https://andruwik777.github.io/JuggleJs

## Demo

<p align="center">
  <img src="./demo.gif" width="45%" style="display:inline-block; margin-right: 10px;"/>
</p>

## Run locally

Clone the repo and open `index.html` in Chrome or Firefox (or use a local server so the TFLite model loads correctly).

## Video test (pure JS, no framework)

To assert juggle count on a fixed MP4 file:

1. Put your video in the repo root as `test.mp4` (or set `VIDEO_URL` in the test page).
2. In `test-juggle-video.html`, set `EXPECTED_JUGGLES` to the expected count for that video.
3. Serve the repo from root (e.g. `npx serve .` or open via GitHub Pages) and open `test-juggle-video.html` in the browser.  
   (Opening the file directly with `file://` may fail due to CORS/model loading; a local server is recommended.)
4. The page will run the video, run the same detection loop as the live app, and show **PASS** or **FAIL: expected X, got Y** in the top-left corner when the video ends.

**Debug:** Set a breakpoint in DevTools (e.g. in `predictWebcam` or `displayVideoDetections` in `js/app.js`). When execution pauses, the video pauses too (same main thread), so you see the exact frame being processed.

<p align="center">
  <b>Have fun juggling and go break some personal records! ⚽️🏆</b>
</p>
