# regex tab

most used tab, by now the most overloaded. read this before poking buttons.

the whole thing runs off a number pulled from each filename. you give it a regex with one
capture group (default `_(\d+)ml`), it grabs that number from every file, sorts the curves by
it and colors them viridis (dark = low, yellow = high). everything past that is a question
about the whole ordered family at each frequency, not about any one curve. so "monotonic"
means the curves keep their order as the number grows, not that a single curve goes up. files
that dont match the pattern get skipped with no warning. group picks which capture group if
you wrote more than one.

the colored stripes, each one a different question:

- green, monotonic: curves stay in order as the param grows. `strict` = no ties. `drop N`
  brute-forces tossing the N worst curves to widen the green (dropped ones go dashed). caps at n-2.
- blue, small diffs: inside green, where the curves bunch within `tol` (scaled by the `@ X dB` level).
- orange, kendall tau: rank agreement at or above the threshold. softer monotonic.
- purple, max disp: no curve more than N places out of its expected rank.
- magenta, shared shape.
- cyan, track shift.

shared shape (magenta band + dashed line) means the curves are the same shape, just shifted;
the dashed black line is the shape it found. `vert` is same shape stacked at different dB levels
(parallel curves). `freq-shift` is same shape but slid sideways in frequency first, then compared.
`congr` is how identical they have to be, 0 to 1, drop it for noisy data. `window` is how wide a
chunk it looks at, in samples not GHz.

track shift (cyan band + side plot) is the quantitative one. it finds where the curves are one
feature sliding with the param, a notch walking down in frequency as volume goes up say, and then
measures it: the side panel plots shift in MHz against your regex number with a line fit, slope in
MHz per unit plus R^2. `max shift` is the biggest slide to look for, in samples, and the window
auto-sizes off it. `mono` is how cleanly the slide has to march with the param, 1.0 strict, lower
tolerates a scrambled slide. `congr` is shared with shared shape.

gotchas, the counterintuitive bits:

- window and max shift are in samples, not frequency. tune them against your sweep point count.
- track shift finds nothing if max shift is smaller than the real slide. bump it.
- over a short window a small frequency shift reads as a vertical offset, so vert flags slides
  too. widen the window to tell vert and freq-shift apart.
- `congr` is shared by shared shape and track shift, one knob for both.
- gating only works if you launched through `mainscript.py`, not ui_main.py. the time-gate patch
  lives in mainscript.
- magnitude is dB when the global "dB scale" box is on, else linear. the phase box overrides the
  lot and gives unwrapped degrees.

## how each method actually works

monotonic: at every frequency it lines the curves up in regex order and looks at the gaps between
neighbours. if every gap points the same way (all up the stack or all down) the curves are in
order there and it goes green. strict just forbids exact ties.

drop N: same test, but it tries every way of removing N curves and keeps whichever removal leaves
the most frequencies monotonic. stops one or two badly-behaved files poisoning the whole band.

kendall tau: monotonic but graded instead of pass/fail. at each frequency it takes every pair of
curves and asks whether the higher-regex one sits above the lower one (concordant) or below
(discordant). tau = (concordant - discordant) / total pairs, so +1 is perfect order, 0 is a
coin-flip, -1 is fully reversed. a few inversions just dent it instead of killing the band, which
is why its softer than the green test.

max disp: rank the curves by value at each frequency, compare to where they ought to sit by regex
order, take the worst single curve's displacement. threshold N means no curve is more than N
places out of line. catches a local swap that the tau average would smear over.

small diffs: only looks inside green regions. at each frequency it checks the neighbour gaps and
flags where enough of them fall under tol (tol scaled by the level you set in @ X dB, so it adapts
to whether youre sitting at -10 or -30 dB). marks where the family is bunched, not just ordered.

congruence (shared shape): chop out a window, subtract each curve's own mean so only shape is
left, then split the leftover wiggle into the part common to all curves and the part unique to
each. congr is the common fraction, 1 = identical shape. vert does this on the raw window.
freq-shift slides each curve sideways to best line them up first, so it catches the same shape
sitting at a different frequency.

track shift: finds the window where the curves are one feature sliding in step with the param,
then pins the shift by cross-correlating each curve against the middle one to get its lag in
samples, converts that to MHz, and least-squares fits shift against your regex number. slope is
your MHz-per-unit, R^2 says how linear it actually is.
