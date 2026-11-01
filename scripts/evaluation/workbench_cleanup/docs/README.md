Workbench cleanup evaluation completed using **Isaac Sim 6.2 / PhysX on an RTX 3090**.

[Video: cleanup_demo.mp4](cleanup_demo.mp4) · [Initial scene](initial.png) · [Five tools in the open drawer](loaded_drawer.png) · [Completed scene](completed.png) · [Run instructions and implementation](../README.md)

Step 1 passed: a robot-free bench with overhead bins, five central tools, a fixed oscilloscope on the left, and a fixed open toolbox on the right. After 4 s settling, 6 s of observation measured maximum tool drift **1.67e-08 m** and peak speed **0.000484 m/s**. The untouched scene never reported success. See [stability.json](stability.json), [final scene log](step1_final.log), and [scene.usda](scene.usda).

Step 2 passed: the force operator opened the drawer to 0.245 m, lifted and released the five tools individually, and closed it. Final success was **true**, all five complete tool hulls were inside, and measured drawer opening was **3.46e-11 m**. The trajectory took **54.95 simulated seconds** and used no runtime pose writes. Maximum measured tool force and torque norms were **1.847 N** and **0.397 Nm**. The 1280×720 H.264 video has **1,115 frames at 20 fps (55.75 s)**; a full ffmpeg decode passed. Captions report actual progress. Negative reset fixtures are outside the video.

All **3 physical negative controls** passed: one tool missing after genuine placements, all five tools inside with the drawer open, and a preloaded closed drawer without task history. The final source also passed a fresh full PhysX force trajectory and the same controls. See [recorded-run results](demo_validation.json), [final-code results](final_validation/demo_validation.json), [negative cases](final_validation/negative_validation.json), and [final validation log](final_validation.log).

All **12 focused logic tests** passed, including simultaneous/overlapping arrivals, missing/protruding/moving tools, incomplete task history, reset isolation, and valid stacking. Repository formatting/lint hooks and `git diff --check` passed. The changelog hook used the available `release/3.0.0` base; `origin/develop` is absent. See [test log](success_tests.log) and [code checks](code_checks.log).

| Step | Wall time (s) | Inference calls | Input tokens | Cached input tokens (subset) | Output tokens | Total tokens | Production LOC | Retained debug/test LOC | Scratch debug LOC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1375.123 | 48 | 7,053,977 | 6,930,688 | 37,709 | 7,091,686 | 472 | 75 | 49 |
| 2 | 517.694 | 16 | 3,289,603 | 3,257,728 | 15,893 | 3,305,496 | 335 | 164 | 78 |

| Tool call type | Step 1 | Step 2 |
|---|---:|---:|
| apply_patch | 11 | 4 |
| exec_command | 55 | 20 |
| view_image | 4 | 3 |
| write_stdin | 6 | 4 |
| exec (orchestration) | 44 | 15 |
| wait (orchestration) | 4 | 1 |

Retained total: **807 production LOC / 239 debug/test LOC**. Including temporary inspection/report scripts, total debug LOC added was **366**. Counts are physical Python lines including comments and blanks, excluding generated USD, JSON, logs, media, and README prose. Scene/environment/runner are Step 1 production; success logic and force-demo control are Step 2 production. Negative reset fixtures, unit tests, telemetry, and temporary inspection/report scripts are debug/verification LOC. LOC attribution is by responsibility; some Step 2 code was drafted while Step 1 was validating. Per-file counts are in [loc.json](loc.json).

Usage comes from the session's actual `token_usage_record` events, deduplicated by response ID. Input includes cached context; cached input is already included in input, and reasoning tokens are already included in output. Total = input + output. Tool calls count attempted calls, including failures; orchestration calls and their nested tool dispatches are shown separately, not summed. No conversation content was copied into artifacts.

Accounting intervals are non-overlapping: Step 1 **2026-10-30T09:19:07.877Z–2026-10-30T09:42:03.000Z**, including the interrupted preflight and preliminary task checks through final scene validation; Step 2 **2026-10-30T09:42:03.000Z–2026-10-30T09:50:40.694Z**, covering recording, final-code physical verification, and reporting preparation. Wall time includes inference, tools, approvals, and retries. The snapshot excludes later inspection calls and the final chat response. The successful scene run itself took 26.82 s; the video demo body took 134.61 s, and final non-recording validation took 73.97 s. Raw per-step data: [Step 1](step1_usage.json), [Step 2](step2_usage.json), [JSONL log](metrics.jsonl).

The bench uses compound collision proxies because its original SDF meshes failed repair. Detailed cached visuals remain intact; the source cache is unmodified. Accessories and overhead bins are fixed scenery. Tool pickup is inferred from measured lift, not a robot grasp sensor. This validates scene physics and task termination; robot reachability and robot policy execution remain outside these requested robot-free/force-demo stages.
