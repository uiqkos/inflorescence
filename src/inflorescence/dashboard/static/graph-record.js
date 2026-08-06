/* Screen-recorder for the graph <canvas> — lets the replay be saved as a video for
 * READMEs/demos. No deps, no modules: plain global, matches the rest of the dashboard. */
"use strict";

(function () {
  // MP4/H.264 first: it drops straight into GitHub, Slack and QuickTime, while a
  // WebM has to be converted first. Note the codec string — Chrome rejects the
  // "h264" alias and only recognises the full avc1 profile id; Safari accepts the
  // bare "video/mp4". WebM stays as the fallback for browsers that record nothing
  // else (Firefox, older Chrome).
  // avc3 before avc1: it carries the parameter sets in-band, so resizing the window
  // mid-take just changes the picture instead of corrupting the stream (Chrome warns
  // about exactly this when avc1 is used).
  const MIME_CANDIDATES = [
    "video/mp4;codecs=avc3.4d002a",  // H.264 main profile
    "video/mp4;codecs=avc3.42E01E",  // H.264 baseline
    "video/mp4;codecs=avc1.4d002a",
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];

  // Recorder state lives in one place so a failed/aborted recording always
  // resets cleanly and a later start() is never left stuck thinking it's busy.
  let rec = null;
  let stream = null;
  let chunks = [];
  let startedAt = 0;
  let maxTimer = null;
  let pendingOnStop = null;
  let pendingOnError = null;

  function supportedMimeType() {
    if (typeof window === "undefined" || !window.MediaRecorder || !window.MediaRecorder.isTypeSupported) return null;
    for (const m of MIME_CANDIDATES) {
      if (window.MediaRecorder.isTypeSupported(m)) return m;
    }
    return null;
  }

  function isSupported() {
    if (typeof document === "undefined" || typeof window === "undefined") return false;
    if (typeof HTMLCanvasElement === "undefined" || !HTMLCanvasElement.prototype.captureStream) return false;
    if (!window.MediaRecorder) return false;
    return !!supportedMimeType();
  }

  function pad2(n) {
    return String(n).padStart(2, "0");
  }

  function timestamp() {
    const d = new Date();
    return `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}-${pad2(d.getHours())}${pad2(d.getMinutes())}`;
  }

  function teardown() {
    if (maxTimer) { clearTimeout(maxTimer); maxTimer = null; }
    if (stream) { stream.getTracks().forEach((t) => t.stop()); } // leaked tracks keep the tab's recording indicator lit
    stream = null;
    rec = null;
    chunks = [];
    pendingOnStop = null;
    pendingOnError = null;
  }

  function active() {
    return !!rec && rec.state !== "inactive";
  }

  function elapsedMs() {
    return active() ? performance.now() - startedAt : 0;
  }

  function finish() {
    const mimeType = rec ? rec.mimeType : (supportedMimeType() || "video/webm");
    const ext = /mp4/i.test(mimeType) ? "mp4" : "webm";
    const blob = new Blob(chunks, { type: mimeType });
    const url = URL.createObjectURL(blob);
    const filename = `${pendingOnStop && pendingOnStop.name || "graph"}-${timestamp()}.${ext}`;
    const durationMs = elapsedMs() || performance.now() - startedAt;

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();

    const onStop = pendingOnStop && pendingOnStop.cb;
    teardown();
    // Give the download a few seconds to actually start reading the blob before
    // the URL is revoked — revoking immediately can race the browser's fetch of it.
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    if (onStop) onStop({ blob, url, filename, bytes: blob.size, durationMs });
  }

  function start(canvas, opts) {
    // The browser treats videoBitsPerSecond as a hint and overshoots it on a busy
    // frame, so 4 Mbps measures out at ~6 and keeps a ~10 s Retina take inside
    // GitHub's 10 MB attachment limit.
    const { fps = 60, bitrate = 4e6, name = "graph", onStop, onError, maxMs = 3 * 60 * 1000 } = opts || {};
    if (active()) { onError && onError("A recording is already in progress."); return false; }
    if (!isSupported()) { onError && onError("This browser can't record canvas video (no captureStream/MediaRecorder)."); return false; }
    if (!canvas || typeof canvas.captureStream !== "function") { onError && onError("No canvas to record."); return false; }

    const mimeType = supportedMimeType();
    try {
      stream = canvas.captureStream(fps);
      rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: bitrate });
    } catch (err) {
      teardown();
      onError && onError(`Could not start recorder: ${err.message || err}`);
      return false;
    }

    chunks = [];
    pendingOnStop = { cb: onStop, name };
    pendingOnError = onError;
    rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    rec.onerror = (e) => {
      const msg = (e.error && e.error.message) || "recording error";
      const cb = pendingOnError; // teardown() clears it, so grab it first
      teardown();
      cb && cb(msg);
    };
    rec.onstop = finish;

    startedAt = performance.now();
    rec.start();
    // Belt-and-suspenders cap: a forgotten recording must not eat gigabytes of RAM.
    maxTimer = setTimeout(() => stop(), maxMs);
    return true;
  }

  function stop() {
    if (!active()) return;
    try { rec.requestData(); } catch { /* some UAs throw if already flushing; ignore */ }
    rec.stop();
  }

  globalThis.GraphRecorder = { isSupported, mimeType: supportedMimeType, start, stop, active, elapsedMs };
})();
