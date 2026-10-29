import { useRef, useState } from 'react';

/* Long-form videos: the narrated intro on Overview and a recorded session on Live Demo.
   Both are re-encoded web copies of media/video/* (H.264, faststart) living in public/media. */

export function IntroVideo() {
  return (
    <section className="story-section video-section section-shell" id="intro-video" aria-labelledby="intro-video-h">
      <div className="story-head">
        <span className="eyebrow">Introduction</span>
        <h2 id="intro-video-h">Meet SignBridge.</h2>
        <p>What we are building, and who it is for. A 20-second film with sound.</p>
      </div>
      <div className="video-frame is-wide">
        <video className="video-player" controls playsInline preload="metadata" poster="/media/intro.jpg" aria-label="SignBridge introduction video">
          <source src="/media/intro.mp4" type="video/mp4" />
        </video>
      </div>
    </section>
  );
}

const CHAPTERS = [
  { t: 0, label: 'Start the camera' },
  { t: 10, label: 'Sign “look”' },
  { t: 20, label: 'Sign “wolf”' },
  { t: 24, label: 'Sign “shhh”' },
  { t: 31, label: 'Finalize → spoken sentence' },
];

const mmss = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;

export function DemoRecording() {
  const ref = useRef<HTMLVideoElement | null>(null);
  const [now, setNow] = useState(0);
  const active = CHAPTERS.reduce((acc, c, i) => (now >= c.t ? i : acc), 0);

  const seek = (t: number) => {
    const v = ref.current;
    if (!v) return;
    v.currentTime = t;
    void v.play().catch(() => undefined);
  };

  return (
    <section className="demo-recording" id="demo-recording" aria-labelledby="demo-recording-h">
      <div className="video-frame">
        <video
          ref={ref}
          className="video-player"
          controls
          muted
          playsInline
          preload="metadata"
          poster="/media/demo-session.jpg"
          onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)}
          aria-label="Screen recording of a live SignBridge session"
        >
          <source src="/media/demo-session.mp4" type="video/mp4" />
        </video>
      </div>
      <div className="demo-recording-copy">
        <span className="eyebrow">Recorded session</span>
        <h3 id="demo-recording-h">No camera? Watch one run first.</h3>
        <p>Three signs become “Look over there, a wolf! Shhh!” in about 35 seconds. Jump to any step:</p>
        <ol className="video-chapters">
          {CHAPTERS.map((c, i) => (
            <li key={c.t}>
              <button type="button" className={i === active && now > 0 ? 'is-active' : ''} onClick={() => seek(c.t)}>
                <span>{mmss(c.t)}</span>{c.label}
              </button>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
