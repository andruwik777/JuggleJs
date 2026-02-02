# Juggle Counter (PoC)

Web app that counts football juggles using the device camera and MediaPipe Object Detection with a custom TFLite model.

Inspired by https://github.com/Logan1904/JuggleNet.git

## Run locally

Clone the repo and open `index.html` in Chrome or Firefox (or use a local server so the TFLite model loads correctly).

## Deploy on GitHub Pages

1. Push your code to a public GitHub repository.
2. In the repo go to **Settings** → **Pages**.
3. Under **Build and deployment** → **Source** choose **Deploy from a branch**.
4. Under **Branch** select `main` (or your default branch) and folder **/ (root)**.
5. Click **Save**. After a minute or two the site will be live at:
   - `https://<your-username>.github.io/<repo-name>/`

Each push to the selected branch will trigger a new deployment automatically. Camera access requires HTTPS; GitHub Pages serves over HTTPS by default.