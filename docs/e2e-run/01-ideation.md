# Ideation — minimal torch-only DDIM-variant scheduler with configurable `prediction_type`

Stage: ideate (PM). Date: 2026-06-21 (round r3). Topic: deliver the golden task as a ranked, grounded shortlist.

**Golden task:** "Add a minimal torch-only scheduler: a DDIM variant with a configurable `prediction_type`."

## Grounding (authority cited by role, not restated)

- **Philosophy** (`PHILOSOPHY.md` → `docs/source/en/conceptual/philosophy.md`): Diffusers "prefer[s]
  copy-pasted code over hasty abstractions" and keeps "an extremely low level of abstraction and very
  self-contained code for pipelines and schedulers" (`:39-40,44`). This is the **single-file policy** —
  "almost all of the code of a certain class should be written in a single, self-contained file" (`:47`) —
  followed "for pipelines, schedulers, and models alike" (`:50`). To add something similar to an existing
  component, "copy the existing file as a starting point and adapt it [and] use `# Copied from` annotations
  on layers that remain identical so `make fix-copies` keeps them in sync" (`:93`).
- **Exemplar (closest prior art):** `src/diffusers/schedulers/scheduling_ddim.py` —
  `DDIMScheduler(SchedulerMixin, ConfigMixin)` (`:139`); `@register_to_config __init__` (`:193`) carrying
  `prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon"` (`:204`); the three-branch
  `prediction_type` dispatch inside `step` (`:455-466`); and the `# Copied from
  ...scheduling_ddpm.DDPMSchedulerOutput with DDPM->DDIM` output dataclass `DDIMSchedulerOutput` (`:31-33`).
  Already torch-only — imports `ConfigMixin`/`register_to_config` (`:25`) and `SchedulerMixin` (`:28`), no
  flax in the core file.
- **Convention surface:** the scheduler subsystem. Registration is two-file and doubled within each file:
  lazy `_import_structure` at `src/diffusers/schedulers/__init__.py:46` plus the `TYPE_CHECKING` import at
  `:154`; top-level export at `src/diffusers/__init__.py:393` plus the re-export at `:1250`. A torch-only
  scheduler also needs a stub in `src/diffusers/utils/dummy_pt_objects.py`. Tests subclass
  `SchedulerCommonTest` (`tests/schedulers/test_schedulers.py`); the exemplar test
  `tests/schedulers/test_scheduler_ddim.py:7` declares `class DDIMSchedulerTest(SchedulerCommonTest)` with
  `scheduler_classes = (DDIMScheduler,)` (`:8`) and already exercises `test_prediction_type`.
- **`.ai/` note:** `.ai/` has `models.md`, `modular.md`, `pipelines.md`, `review-rules.md`, but **no
  dedicated schedulers doc**; `PHILOSOPHY.md` is the governing authority for this subsystem. Tag for
  `improve`: **CANDIDATE-RULE — "new scheduler = copy the nearest `scheduling_*.py` + keep `# Copied from`
  + dual-`__init__` registration (lazy + TYPE_CHECKING, both files) + `dummy_pt_objects` stub +
  `SchedulerCommonTest` subclass."** This is an unwritten, mechanically-checkable convention worth
  codifying.

## Ranked options

Schema per survivor: **value · size · convention-surface · risk** (+ `file:line` anchor).

### 1. RECOMMENDED — Copy `scheduling_ddim.py`, rename, expose `prediction_type` as-is

Copy the exemplar to a new self-contained file (e.g. `scheduling_ddim_variant.py`), rename the class and the
output dataclass, keep the existing `Literal["epsilon", "sample", "v_prediction"]` config field and the
three-branch dispatch verbatim, retain the `# Copied from` annotation on the output dataclass so
`make fix-copies` keeps it in sync, and register in both `__init__.py` files (lazy + TYPE_CHECKING) plus a
`dummy_pt_objects` stub. Test = a new `SchedulerCommonTest` subclass mirroring `test_scheduler_ddim.py`,
including a `test_prediction_type` loop.

- **value:** delivers the literal ask (a DDIM variant whose `prediction_type` is configurable) with maximum
  fidelity to how every existing scheduler is built; zero new concepts for reviewers or users.
- **size:** S — one new file copy-adapted from `scheduling_ddim.py:1-595`, four registration edits
  (`schedulers/__init__.py:46,154`; `__init__.py:393,1250`), a `dummy_pt_objects.py` stub, and one new test
  file mirroring `test_scheduler_ddim.py`.
- **convention-surface:** exactly the established scheduler surface — single-file policy, `# Copied from`,
  dual-`__init__` registration, dummy stub, `SchedulerCommonTest`. Touches nothing else.
- **risk:** Low. Every failure mode is mechanical and caught by an existing gate: forgotten registration
  (import test), broken `# Copied from` (`make fix-copies` / `make quality`), missing dummy object
  (`check_dummies`). No algorithmic novelty to get wrong.
- **anchor:** `src/diffusers/schedulers/scheduling_ddim.py:139,193,204,455-466`.

**Why over others:** it is the copy-adapt-a-real-scheduler path the single-file philosophy explicitly
prescribes (`philosophy.md:39-47,93`) — it reuses a proven torch-only file, introduces no abstraction, and
its only risks are mechanical checks the `make quality` CI gate already enforces. Option 2 adds an
abstraction the philosophy forbids; Option 3 mutates a shared, heavily-referenced file; Option 4
under-delivers the "variant" framing of the task.

### 2. New `BaseDDIMScheduler` + thin subclass (shared base for prediction_type) — CUT

Factor a base class holding the `prediction_type` dispatch, with the new scheduler subclassing it.

- **value:** marginal DRY appeal only.
- **size:** M — new base + subclass + refactor pressure on the existing `DDIMScheduler`.
- **convention-surface:** introduces a cross-scheduler abstraction layer that does not exist today.
- **risk:** High — **directly violates** the single-file / copy-paste-over-abstraction policy
  (`philosophy.md:39-47`); a philosophy-fit reviewer would reject on sight.
- **CUT REASON:** new abstraction is exactly what the library's philosophy forbids for schedulers.

### 3. Add a `prediction_type` mode to `DDIMScheduler` in place (no new file) — CUT

Extend the existing `DDIMScheduler` rather than adding a variant.

- **value:** smallest diff in isolation.
- **size:** S, but mutates a shared file.
- **convention-surface:** edits `scheduling_ddim.py`, a class referenced across pipelines and `# Copied
  from` chains.
- **risk:** Medium-High — `DDIMScheduler` already supports a configurable `prediction_type`
  (`scheduling_ddim.py:204,455-466`), so this is **redundant**; touching it risks cascading `# Copied from`
  breakage. Also fails the "add a *variant*" framing of the task.
- **CUT REASON:** redundant with existing behavior and risks the shared exemplar; the task asks for a new
  scheduler, not a mutation.

### 4. Bare nn-free function/util implementing the step (no SchedulerMixin) — CUT

A minimal standalone function instead of a `SchedulerMixin`/`ConfigMixin` class.

- **value:** superficially "minimal".
- **size:** S.
- **convention-surface:** abandons the scheduler contract (`register_to_config`, loadable config,
  `SchedulerOutput`, `SchedulerCommonTest`).
- **risk:** High — not loadable/savable, not a drop-in scheduler, untestable via the common harness;
  diverges from every scheduler in the tree.
- **CUT REASON:** "minimal" means minimal *within* the scheduler contract, not bypassing it; the result
  would not be a scheduler at all.

## Recommendation

**Option 1 — copy-adapt `scheduling_ddim.py` into a new self-contained DDIM-variant scheduler** with a
configurable `prediction_type`, registered in both `__init__.py` files (lazy + TYPE_CHECKING) with a
`dummy_pt_objects` stub, and covered by a `SchedulerCommonTest` subclass mirroring `test_scheduler_ddim.py`.

## Next step

Route Option 1 into `brainstorm` to pin the exact new file/class/output-dataclass names and confirm the
`# Copied from` boundaries (which lines stay copy-tracked vs. rename-adapted), then `plan`.
