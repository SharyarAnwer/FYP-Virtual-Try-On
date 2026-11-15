# Photo Try-On

A second way to try clothes on, beside the live camera overlay: eight photos of the wearer, an outfit
chosen from recommendations, and back comes a turntable of the wearer dressed in it - real photos,
one every 45 degrees round them, turned by dragging a finger.

It is opened from the **Photo try-on** button on the live try-on's main screen. Its code is in
`app/src/main/java/com/shahryar/virtualtryon/phototryon/`.

## How it works

| Step | Where | What happens |
|---|---|---|
| 1. Capture | Phone | `GuidedCaptureActivity` talks the wearer into place and takes the front photo hands-free, then picks seven more views out of one slow turn. `CaptureGuide` decides each instruction and when each view is taken. |
| 2. Measure | Phone | `PhotoAnalyzer` runs the pose and face models on the photos. `BodyMeasure` gives height, shoulders, chest, waist, hips, arm and leg; `SkinTone` gives skin depth and undertone; the live try-on's `FaceShapeMeter` gives face shape. |
| 3. Recommend | Phone | `PhotoRecommender` ranks every shirt with every pair of trousers, using the live try-on's `Recommender` unchanged plus rules for skin tone, measured build and size. |
| 4. Dress | Modal | The engine (`modal/engine.py`) dresses the eight views on Modal's GPUs: **Leffa** for seven of them, **FASHN VTON v1.5** for the straight back, which Leffa drew wrong in every test. |
| 5. Show | Phone | `TryOnResultActivity` shows each view as it arrives and keeps a copy, so an outfit seen once opens again with no engine. |

Scale comes from the eye: an adult's iris is 11.7 mm across, so the wearer's eye spacing is measured
while they are at the phone, then found again in the full-body photo. Nothing is typed.

## The engine

The engine runs on Modal and is always available at a fixed address; nothing has to be started by
hand. Modal starts GPUs when a request comes and stops them when idle, and bills by the second.

- First time only: `python -m modal run engine.py::fetch_weights` (puts Leffa's files in a Modal volume).
- After any change to `modal/engine.py`: `python -m modal deploy engine.py`.
- The address for the app: `python engine_key.py`. Paste it once on the **Your profile** screen.
- A check that puts one outfit through it as the app does: `python engine_check.py <photos folder> <output folder>`.

The Kaggle notebooks in `kaggle/` are the earlier version of the engine, kept for reference.

## How long it takes

Measured on Kaggle (two T4 cards) with eight views, navy polo and khaki chinos, before the back-view
retries were added: front 1 min 13 s, back 1 min 42 s, both sides 2 min 53 s, all eight 5 min 14 s.

## Known limits

- The dressing model sometimes paints the wrong shirt on views of the back. The engine measures this,
  tries again with another model or seed, and leaves a view out of the turntable if every try is wrong.
- The views of the turn are taken from the camera's video while the wearer moves, so they can be softer
  than the posed front photo, and the side measurements are rougher for it.
- Measurements are estimates: about one part in twenty from the scale, more for chest, waist and hips,
  and loose clothes measure bigger than the body under them. Skin tone depends on the light.
- The dressing model guesses the body under loose clothes, so the wearer can come out slimmer, and it
  sometimes invents a wristwatch or shoes. It does not know sizes.
- The grey tee and the white shirt have no back picture, so their back views use the front one.
- One person can use the engine at a time, and only while its Kaggle session is running.

## Licences

Leffa is MIT-licensed. The pictures of garments are listed in `app/src/main/assets/garments/real/SOURCES.md`.
