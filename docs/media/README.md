# Simulation and hardware media

The root README presents three simulation recordings and one hardware recording in a two-by-two gallery, with animated highlights and full-video links. The clips keep their original aspect ratios.

| Recording | Repository file | Full download | Duration |
| --- | --- | --- | ---: |
| MuJoCo simulation | [simulation.mp4](simulation.mp4) | [Release video](https://github.com/topsun-bot/lingxi-x2-ppo/releases/download/v0.1.0/simulation.mp4) | 35 s |
| X2 hardware experiment | [hardware.mp4](hardware.mp4) | [Release video](https://github.com/topsun-bot/lingxi-x2-ppo/releases/download/v0.1.0/hardware.mp4) | 20.434 s |
| Earlier full-speed simulation | [simulation-full-speed.mp4](simulation-full-speed.mp4) | [Release video](https://github.com/topsun-bot/lingxi-x2-ppo/releases/download/v0.1.0/simulation-full-speed.mp4) | 35 s |
| Earlier low-speed simulation | [simulation-low-speed.mp4](simulation-low-speed.mp4) | [Release video](https://github.com/topsun-bot/lingxi-x2-ppo/releases/download/v0.1.0/simulation-low-speed.mp4) | 31 s |

The simulation shows the released History2 Hybrid policy and is described in the [model provenance](../provenance.md). The hardware recording was supplied for publication on 2026-10-27. It shows an X2 in a test-rig environment and does not identify its checkpoint or deployment revision; it is separate demonstration footage, not additional validation of the released model or the public ROS adapter.

The two additional simulations are archived experiments from the same training project, using earlier 104-input policies with the Kp090/Kd080 gain recipe. They are not recordings of the released 208-input History2 Hybrid. The full-speed clip comes from the `x2_ultra_full_speed_final_v1` artifact collection, whose delivery notes identify `model_7450.pt`. The low-speed clip comes from `x2_ultra_kp090_kd080_from_scratch_v1`, whose delivery notes identify `model_final.pt`. These checkpoint associations follow the archived delivery notes, not independent metadata embedded in the video. The low-speed candidate passed nominal evaluation but did not pass its robust evaluation; these clips add demonstrations, not new acceptance claims. Historical checkpoints for these two clips are not bundled with this release.

Both additional MP4s were copied without re-encoding. The full-speed file is 960 × 540; the low-speed file is 1280 × 720; both are H.264 at 25 fps without audio. Their GIFs contain three four-second highlights at 7 fps and 640 × 360. Full-speed highlights use seconds 4–8, 15–19 and 28–32; low-speed highlights use 4–8, 14–18 and 23–27.

The hardware MP4 retains the uploaded H.264 video, AAC audio and 720 × 1280 portrait framing. Container metadata was removed and the file was remuxed for progressive playback without re-encoding the audiovisual streams. Its animated preview uses seconds 2–14 at 8 fps, resized to 270 × 480. The simulation preview uses three four-second highlights. Both GIF previews are silent and last approximately 12 seconds.

Public hardware MP4 SHA-256:

```text
4845b1a794b3fd2708d18244962072baf93c9c3b349c9cb1f9f247be7450f8c2
```

Hardware GIF SHA-256:

```text
9e3e2dc5bb1da62baf8b6c477f623343487298fa5bbb0c0c005c17967cbc60e7
```

Additional archived MP4 SHA-256:

```text
90472692f148755bbb3db3f8d23bd8d840b10540ab824a5228663dd001e976d6  simulation-full-speed.mp4
f2b389f1d7a0ee2fd565801f58b36c38dc32f15d9585607940adf05eb4aa378a  simulation-low-speed.mp4
```
