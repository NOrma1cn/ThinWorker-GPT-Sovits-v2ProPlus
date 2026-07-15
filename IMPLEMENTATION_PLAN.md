## Stage 1: Audit Document And Baseline
**Goal**: Record the complete inference audit, evidence, hypotheses, and ordered validation gates.
**Success Criteria**: The repository contains one review document that distinguishes confirmed behavior from unverified opportunities; the current unit-test baseline is recorded.
**Tests**: `PYTHONPATH=src python -m pytest -q` passes.
**Status**: Complete

## Stage 2: PCM Conversion Correctness
**Goal**: Verify and remove the positive full-scale PCM16 wraparound that can create single-sample clicks.
**Success Criteria**: `+1.0` maps to `+32767`, out-of-range samples are clipped safely, and existing streaming splice behavior remains unchanged.
**Tests**: Add focused PCM conversion tests; run the complete unit-test suite; generate listening samples if the production WSL environment is available.
**Status**: Complete

## Stage 3: Exact-Equivalence Frontend Cleanup
**Goal**: Bypass language detection for conservatively recognized Chinese-only input and remove unused MLM-head computation without changing normalized text, phones, G2PW output, or RoBERTa hidden features.
**Success Criteria**: Golden frontend outputs are identical on representative Chinese, numeric, punctuation, polyphonic, and mixed-ASCII inputs; pure-Chinese startup no longer loads the language detector; stage latency decreases.
**Tests**: Golden-output tests, model hidden-state parity test, cold/warm frontend benchmark.
**Status**: Complete

## Stage 4: Streaming Runtime Experiments
**Goal**: Validate RNG isolation, `first=50 / steady=120`, cached VITS static conditioning, static-shape VITS execution, and bounded semantic context independently.
**Success Criteria**: Each experiment has separate latency, PCM/spectral similarity, boundary metrics, and listening artifacts; only experiments passing their stated gates enter production.
**Tests**: Opening/long benchmarks over multiple seeds, semantic-token hashes, boundary-jump checks, generated HTML listening set.
**Status**: Complete

## Stage 5: Voice Profile And Model-Family Benchmark
**Goal**: Validate offline fixed-voice profiles and compare the optimized chain with at least one modern Chinese streaming TTS family.
**Success Criteria**: Voice-profile cache invalidation is fingerprinted and output-equivalent; challenger comparison uses identical hardware, texts, reference voice, and user listening gates.
**Tests**: Profile parity and invalidation tests; TTFB/RTF/VRAM/CER/speaker-similarity benchmark; blind listening report.
**Status**: In Progress

## Stage 6: Buffer-Aware Mode 4 Scheduler
**Goal**: Replace Mode 4's repeated fixed 50-token cap with a playback-buffer-aware deadline while preserving explicit legacy schedules.
**Success Criteria**: The first chunk remains capped; the cap relaxes above the target buffer and reactivates before predicted underrun; explicit `50/50` and `50/120` requests keep their old behavior.
**Tests**: Pure scheduler state tests, server request-policy tests, and the complete unit-test suite.
**Status**: Complete

## Stage 7: Runtime Latency Validation
**Goal**: Validate the new default on short, medium, and long input under the production Triton/G2PW CUDA profile.
**Success Criteria**: No predicted playback underrun, unchanged first semantic chunk, fewer long-form chunks, and no TTFB regression beyond run variance.
**Tests**: Alternating repeated Mode 4 legacy/adaptive profiling with buffer-margin and semantic-hash checks.
**Status**: Complete

## Stage 8: Listening Gate
**Goal**: Produce a compact legacy/adaptive Mode 4 listening comparison.
**Success Criteria**: User confirms that later speech is stable without new overlap, clicks, drift, or pauses.
**Tests**: HTML listening set for representative medium and long inputs.
**Status**: Complete

## Stage 9: Position-Consistent VITS Noise POC
**Goal**: Reuse the same request-local noise values for acoustic frames that overlap consecutive VITS chunks, without changing the first chunk or the default path.
**Success Criteria**: First-call noise is bitwise equal to the current generator; overlap frames reuse cached noise; new frames preserve the current RNG sequence; invalid frame gaps fail fast.
**Tests**: Focused CPU noise-cache tests and the complete unit-test suite.
**Status**: Complete

## Stage 10: Noise POC Runtime Validation
**Goal**: Compare current and position-consistent noise with identical Mode 4 scheduling, seed, and semantic output.
**Success Criteria**: Identical first package and semantic hashes, unchanged chunk schedule and buffer margin, plus objective seam metrics and latency measurements.
**Tests**: Alternating repeated medium/long benchmark under Triton/G2PW CUDA.
**Status**: Complete

## Stage 11: Noise POC Listening Gate
**Goal**: Determine whether overlap-consistent VITS noise audibly reduces residual distortion.
**Success Criteria**: User listening result decides promotion or rejection; no automatic default change.
**Tests**: Compact current/consistent HTML listening set.
**Status**: Complete
