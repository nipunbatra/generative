# Author review: Lecture 4

20 October 2026. This record distinguishes local calculations, saved measurements and visual review.

- Executed scalar D/G steps in float64; independently checked stable BCE and G finite differences. Both parameter owners are checked explicitly.
- Executed all notebook cells: fresh neural ownership tests, fresh 400-step toy training and saved MNIST checkpoint replay. No unhandled errors; three actual notebook figures inspected.
- Replayed the original 30-epoch GAN at the original fixed noise seed; maximum sample difference 5.28e−6 across device arithmetic.
- Independently rendered and inspected two-player introduction, initial toy density and G-only outcome after reviewer corrections. Intro labels now say learned θ/ω rather than implying simultaneous updates.
- Density comparisons use fixed y limits including the peak. G-only frames use the same [−30,30] x range; the final frame reports the measured 56.6% of samples outside [−5,5]. This makes the broad drift visible rather than hiding it outside the plot.
- Plot legends were moved outside the data region after inspecting peak overlap. Detach diagram and code are separate teaching scenes.
- Independent all-state visual review by the attention benchmark agent supplied the above corrections. Root generates complete browser/PDF QA separately.

This is an authored 60-minute route, not a recorded classroom rehearsal. Full MNIST training was not rerun; checkpoint replay and fresh small updates are labelled separately.
