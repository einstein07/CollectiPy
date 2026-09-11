# Deferred Items -- Quick Task 260911-msr

Observed while running the regression set and the experiment; none touch what this
task changed.

- Pre-existing test failures, identical on the untouched tree (commit cb4a50d with
  the working tree's untracked tests copied in): 7 in `tests/test_bifurcation.py`
  (`AttributeError`, behavioral / gradient / omega paths),
  `tests/test_shared_sensory_stream.py::test_l1_ring_attractor_matches_its_post_seeding_reference`
  (the known "seeded ring attractor drifted" baseline), and the rest of the 15 listed
  in the baseline run. Not investigated.
- `BifurcationDetector.update` in `behavioral` mode dispatches to
  `_update_behavioral_agent_angle` (the bump-angle branch is commented out): an event
  fires when a target bearing is within `alignment_tolerance_deg` of the agent
  heading, i.e. of 0 in the egocentric frame. A target placed dead ahead fires on
  tick 1; a bump parked between two targets never fires. The experiment therefore
  reads commitment from the per-tick logs and keeps the detector's event only for
  cross-reference. The detector is out of scope (spec 4.2).
- The runtime readout (`use_thresholding: false` -> circular mean of the whole ring)
  is what re-merges two bumps into a midpoint heading. A thresholded / peak readout is
  the obvious next lever; readout changes are outside this spec (4.2, 8).
- `Environment.start()` forks agent processes, so in-process trials cannot run in a
  daemonic `multiprocessing.Pool` worker; the local driver uses one subprocess per
  task instead. Worth remembering for any future local sweep driver.
- `reduction: pnorm` is implemented and tested but not part of the design (8).
