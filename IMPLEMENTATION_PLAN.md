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
