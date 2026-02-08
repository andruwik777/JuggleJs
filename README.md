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

https://andruwik777.github.io/JuggleJs/test-juggle-video.html

**Modes:** Use **"Enable DEBUG mode"** (unchecked by default) only when you need to debug: then the test runs frame-by-frame (seek → detect), so breakpoints in `runOneDetectionFrame()` pause the video, but playback is ~0.5x. With DEBUG unchecked, the test runs in real-time (`video.play()` + detection on each frame) for a faster run.

**Debug:** Set a breakpoint in `runOneDetectionFrame()` in `js/app.js` on the line with `objectDetector.detectForVideo(...)`. Only in DEBUG mode will the video pause when execution stops there.

<p align="center">
  <b>Have fun juggling and go break some personal records! ⚽️🏆</b>
</p>
