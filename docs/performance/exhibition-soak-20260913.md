# Exhibition operational test — September 13, 2026

The final run passed 7,350.938 continuous qualifying seconds from 18:44:26 to
20:46:56 Bangkok time. There were 1,471 qualifying five-second samples; the largest
sample gap was 5.016 seconds. No qualifying storage, upload, application or display
fault reset the clean interval.

| Measurement | Result |
| --- | --- |
| Display configuration | Game left, map right; NVIDIA borderless fullscreen |
| Mock visitors | 15, lasting 158–826 seconds, immediate handoffs |
| Verified archives | 35 |
| Verified images / total files | 3,420 / 3,525 |
| Average display / generation FPS | 59.77 / 12.78 |
| Lowest one-minute display / generation FPS | 57.09 / 11.56 |
| Maximum observed GPU temperature | 65 °C |
| Minimum available disk | 64.14 GiB |
| Longest observed upload | 201.3 seconds |

Cloud restore verified the final 89-file archive. An isolated prune test removed a
verified completed copy while retaining the unfinished control. Production originals
were retained. Both independent Windows display outputs stayed active throughout.
Native window visibility, geometry and display topology were sampled; the operator
confirmed both physical screens during preflight. This was not a continuous external
camera recording or a guarantee about future hardware behavior.

Detailed JSON, checksums and reports remain locally under
`C:\Exhibition\Commissioning\soak-20260913\run-184255`. Synthetic visitor input was
disabled at completion. At 20:55:51, after the test, the artwork recorded an Escape
exit with code 0 and the supervisor restarted it. Post-restart display performance
was approximately 34 FPS at that observation, outside the qualifying interval.

Authenticated SSH recovery and Discord agent operation from the other machines
were not verified by this run. The final idle-caption edit was checked separately
with the map rendering regression; the two-hour result predates that cosmetic edit.
