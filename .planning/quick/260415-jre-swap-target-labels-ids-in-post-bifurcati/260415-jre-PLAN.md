---
phase: quick-260415-jre
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/arena.py
  - src/entity.py
  - tests/test_target_swap.py
  - .planning/phases/03-target-swap/03-CONTEXT.md
autonomous: true
requirements: []
must_haves:
  truths:
    - "After _execute_post_bif_swap fires, entity_a.get_name() returns what entity_b.get_name() returned before the swap, and vice versa"
    - "shape.metadata['entity_name'] on each entity stays consistent with get_name() after the swap"
    - "Agents looking up a target by label after the swap find it at the position that label now occupies"
  artifacts:
    - path: src/arena.py
      provides: "_execute_post_bif_swap swaps _entity_uid and shape.metadata['entity_name'] in addition to XY and strength"
    - path: src/entity.py
      provides: "Entity exposes a set_name(uid) method to allow safe mutation of _entity_uid with metadata sync"
    - path: tests/test_target_swap.py
      provides: "test_execute_post_bif_swap_names verifies the name swap"
  key_links:
    - from: src/arena.py
      to: src/entity.py
      via: "entity.set_name(uid)"
      pattern: "set_name"
    - from: src/arena.py
      to: "shape.metadata['entity_name']"
      via: "set_name() keeps metadata in sync"
      pattern: "metadata.*entity_name"
---

<objective>
Extend `_execute_post_bif_swap` in `src/arena.py` so that entity names/IDs (labels) are
also swapped between the two target objects in addition to their XY positions and strength.

Purpose: Agents that track a target by label (e.g. "target_A") continue following the
correct physical target after the swap, because the label moves with the quality/position
that was originally attached to it.

Output: Updated `_execute_post_bif_swap`, a `set_name()` helper on `Entity`, a new
regression test, and updated D-02 in CONTEXT.md.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/03-target-swap/03-CONTEXT.md

<interfaces>
<!-- Key contracts the executor needs. Extracted from src/entity.py and src/arena.py. -->

From src/entity.py — Entity base class:
```python
class Entity:
    _entity_uid: str               # holds the name/label; set in __init__ via _build_entity_uid()
    def get_name(self) -> str:     # returns self._entity_uid
        return self._entity_uid
    # No set_name() yet — must be added (Task 1)
```

From src/entity.py — StaticObject.__init__ (target objects are StaticObject):
```python
self.shape = Shape3DFactory.create_shape(...)
if hasattr(self.shape, "metadata"):
    self.shape.metadata["entity_name"] = self.get_name()  # set at init; must be kept in sync
```

From src/arena.py — _execute_post_bif_swap (current, truncated):
```python
def _execute_post_bif_swap(self, tick: int) -> None:
    event = self._post_bif_swap_event
    objects_by_name = self._index_objects_by_name()
    for left_id, right_id in event.get("pairs", []):
        left_obj = objects_by_name.get(left_id)
        right_obj = objects_by_name.get(right_id)
        ...
        self._swap_object_xy_positions(left_obj, right_obj)
        left_obj.strength, right_obj.strength = right_obj.strength, left_obj.strength
        # NAME SWAP MISSING — add here (Task 2)
```

From src/arena.py — _index_objects_by_name:
```python
def _index_objects_by_name(self) -> dict:
    indexed = {}
    for _, entities in self.objects.values():
        for entity in entities:
            name = entity.get_name() if hasattr(entity, "get_name") else None
            if name:
                indexed[str(name)] = entity
    return indexed
```

From tests/test_target_swap.py — _MockEntity (must grow a set_name() method):
```python
class _MockEntity:
    def __init__(self, name: str, x, y, strength, z=0.0):
        self._name = name
        ...
    def get_name(self) -> str:
        return self._name
    # set_name() must be added here too (Task 3), mirroring Entity.set_name()
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add set_name() to Entity and keep shape.metadata in sync</name>
  <files>src/entity.py</files>
  <behavior>
    - set_name(uid) sets self._entity_uid = uid
    - If self has a shape attribute with a metadata dict, also sets shape.metadata["entity_name"] = uid
    - get_name() after set_name(x) returns x
    - Entities without shape.metadata are unaffected (no AttributeError)
  </behavior>
  <action>
    In `src/entity.py`, add the `set_name` method to the `Entity` class immediately after
    the existing `get_name` method (line ~88):

    ```python
    def set_name(self, uid: str) -> None:
        """Set the entity UID / label and keep shape metadata in sync."""
        self._entity_uid = uid
        shape = getattr(self, "shape", None)
        if shape is not None and hasattr(shape, "metadata"):
            shape.metadata["entity_name"] = uid
    ```

    This is the ONLY change to entity.py. Do not alter _build_entity_uid, get_name,
    or any other method.
  </action>
  <verify>
    <automated>cd /home/sindiso/Documents/PhD/ring-attractor/CollectiPy/.claude/worktrees/naughty-clarke && .venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from entity import EntityFactory
# StaticObject has a shape with metadata
cfg = {'_id':'idle','shape':'sphere','strength':[1.0],'position':[[0,0,0]]}
obj = EntityFactory.create_entity('object_static_sphere',cfg,0)
orig = obj.get_name()
obj.set_name('new_label')
assert obj.get_name() == 'new_label', 'get_name() did not update'
if hasattr(obj,'shape') and hasattr(obj.shape,'metadata'):
    assert obj.shape.metadata.get('entity_name') == 'new_label', 'metadata not updated'
print('set_name OK')
"
    </automated>
  </verify>
  <done>Entity.set_name(uid) exists; get_name() returns uid; shape.metadata["entity_name"] is updated if present.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Swap entity names in _execute_post_bif_swap</name>
  <files>src/arena.py</files>
  <behavior>
    - After swapping XY and strength, left_obj.get_name() returns the original right_id, and right_obj.get_name() returns the original left_id
    - The log message reflects the swap including names
    - The existing position and strength swap behaviour is unchanged
  </behavior>
  <action>
    In `src/arena.py`, inside `_execute_post_bif_swap`, after the existing strength swap
    line (line ~509):

    ```python
    left_obj.strength, right_obj.strength = right_obj.strength, left_obj.strength
    ```

    Add the name swap immediately after (before the logging.info call):

    ```python
    # Swap entity labels so agents tracking by name follow the correct target (updated D-02)
    left_obj.set_name(right_id)
    right_obj.set_name(left_id)
    ```

    Update the logging.info message to say "positions + strength + names" so the log
    reflects the full swap:

    ```python
    logging.info(
        "Post-bifurcation swap executed at tick %s: %s <-> %s (positions + strength + names)",
        tick, left_id, right_id,
    )
    ```

    No other changes to arena.py. Do not alter _apply_target_position_swap_event or any
    other swap path — name swapping applies only to the post-bifurcation dynamic swap.
  </action>
  <verify>
    <automated>cd /home/sindiso/Documents/PhD/ring-attractor/CollectiPy/.claude/worktrees/naughty-clarke && .venv/bin/python -m pytest tests/test_target_swap.py -x -q 2>&1 | tail -20</automated>
  </verify>
  <done>All existing test_target_swap.py tests pass; the name swap is executed in _execute_post_bif_swap.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Add name-swap test and update _MockEntity + CONTEXT.md D-02</name>
  <files>tests/test_target_swap.py, .planning/phases/03-target-swap/03-CONTEXT.md</files>
  <behavior>
    - test_execute_post_bif_swap_names: after swap, entity_a.get_name() == "target_B" and entity_b.get_name() == "target_A"
    - _MockEntity gains a set_name(uid) method that updates self._name
    - D-02 in CONTEXT.md updated to state that labels ARE now swapped
  </behavior>
  <action>
    **tests/test_target_swap.py:**

    1. Add `set_name` to `_MockEntity` (after the existing `get_name` method, ~line 70):
       ```python
       def set_name(self, uid: str) -> None:
           self._name = uid
       ```

    2. Update the docstring list at the top of the file (lines 1-23) to add the new test:
       ```
         test_execute_post_bif_swap_names — names/IDs are exchanged (updated D-02)
       ```

    3. Add the new test function immediately after `test_execute_post_bif_swap_strength`
       (after line ~352):
       ```python
       def test_execute_post_bif_swap_names():
           """Entity labels/IDs are exchanged between target entities (updated D-02)."""
           entity_a = _MockEntity("target_A", x=10.0, y=5.0, strength=2.0)
           entity_b = _MockEntity("target_B", x=-10.0, y=5.0, strength=0.5)
           objects = {"targets": ({"number": 2}, [entity_a, entity_b])}
           dh = _MockDataHandling()
           stub = _make_arena_stub(pairs=[["target_A", "target_B"]], objects=objects, data_handling=dh)
           stub._post_bif_swap_event = {
               "tick": 60,
               "pairs": [("target_A", "target_B")],
               "triggered_by_agent": "agent_0",
               "bifurcation_tick": 50,
           }

           Arena._execute_post_bif_swap(stub, tick=60)

           # Labels move with the swap: each entity now carries the other's original ID
           assert entity_a.get_name() == "target_B", (
               f"Expected entity_a name 'target_B', got '{entity_a.get_name()}'"
           )
           assert entity_b.get_name() == "target_A", (
               f"Expected entity_b name 'target_A', got '{entity_b.get_name()}'"
           )
       ```

    **CONTEXT.md D-02 update:**

    Replace the current D-02 block. Change:
    - Heading: "D-02: What Gets Swapped — Positions AND Quality (intensity)"
      to:       "D-02: What Gets Swapped — Positions, Quality (intensity), AND Labels"
    - Body: "Target labels (IDs) remain fixed — only coordinates and intensity exchange."
      to:   "Target labels (IDs) are ALSO swapped — coordinates, intensity, AND entity
             names exchange. This ensures agents tracking a target by label continue to
             follow the correct physical target after the swap."
    - Update the before/after example to show labels moving:
      ```
      Before swap:
        target_A: pos=(10, 5),  intensity=2.0, name="target_A"
        target_B: pos=(-10, 5), intensity=0.5, name="target_B"

      After swap:
        entity at pos=(-10,5): intensity=0.5, name="target_A"
        entity at pos=(10, 5): intensity=2.0, name="target_B"
      ```
    - Add implementation note: "Name swap uses Entity.set_name(uid) which also updates
      shape.metadata['entity_name'] for consistency with the collision/detection layer."
  </action>
  <verify>
    <automated>cd /home/sindiso/Documents/PhD/ring-attractor/CollectiPy/.claude/worktrees/naughty-clarke && .venv/bin/python -m pytest tests/test_target_swap.py -x -q 2>&1 | tail -20</automated>
  </verify>
  <done>
    All 21 tests in test_target_swap.py pass (20 existing + 1 new name-swap test).
    D-02 in 03-CONTEXT.md states labels are swapped.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| config → arena | swap pair IDs come from JSON config; already validated by _normalize_post_bif_swap_config |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-260415-01 | Tampering | Entity._entity_uid | accept | swap is internal arena logic, no external input reaches set_name() at runtime |
| T-260415-02 | Denial of Service | set_name() on entity without shape | accept | method uses getattr with None guard; no AttributeError possible |
</threat_model>

<verification>
Full test suite:

```bash
cd /home/sindiso/Documents/PhD/ring-attractor/CollectiPy/.claude/worktrees/naughty-clarke
.venv/bin/python -m pytest tests/test_target_swap.py -v 2>&1 | tail -30
```

Expected: 21 tests collected, all pass.
</verification>

<success_criteria>
- `Entity.set_name(uid)` exists and keeps `shape.metadata["entity_name"]` in sync
- `_execute_post_bif_swap` swaps `_entity_uid` on both target objects via `set_name()`
- `test_execute_post_bif_swap_names` passes: after swap, each entity holds the other's original label
- All 20 pre-existing tests in `test_target_swap.py` continue to pass
- D-02 in `03-CONTEXT.md` states labels are swapped
</success_criteria>

<output>
After completion, create `.planning/quick/260415-jre-swap-target-labels-ids-in-post-bifurcati/260415-jre-SUMMARY.md`
</output>
