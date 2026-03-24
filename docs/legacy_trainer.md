# Legacy Go CPU Trainer

We now train NNUE nets using the **Bullet GPU trainer** (Rust, CUDA). The Go-based CPU trainer (`./tuner`) is retained for Texel tuning of eval parameters and for utility functions (selfplay, rescore, shuffle, check-net, convert). It should NOT be used for NNUE training — Bullet is orders of magnitude faster and produces better nets.

## Texel Tuner (still used)

~1268 parameters optimized via Adam. Training data: `.bin` binpack from selfplay, preprocessed to `.tbin` binary cache, disk-streamed. `computeTrace()` mirrors `Evaluate()` recording sparse coefficients. Frozen params: material (coupling), tempo, trade bonuses. PST values are pre-scaled (effective = raw * scale/100).

### Commands

```bash
./tuner selfplay -games 20000 -time 200 -concurrency 6                       # Generate training data (.bin)
./tuner selfplay -games 20000 -time 200 -concurrency 6 -syzygy /path/to/tb  # With Syzygy tablebases
./tuner tune -data training.bin -epochs 500 -lr 1.0                          # Tune eval parameters
./tuner tune -data training.bin -epochs 500 -lr 1.0 -lambda 0.5             # Tune with blended loss (default lambda=0)
```

### Legacy NNUE Training (DO NOT USE — use Bullet instead)

```bash
./tuner nnue-train -data training.bin -epochs 100 -lr 0.01 -output net.nnue  # Train NNUE
./tuner nnue-train -data a.bin,b.bin -epochs 100 -lr 0.01                    # Train from multiple files
./tuner nnue-train -data training.bin -resume net-v1.nnue -epochs 100 -output net-v2.nnue # Resume
```

### Gotchas

- Tuner traces must mirror `Evaluate()` exactly. When modifying eval, update `computeTrace()` to match
- `.tbin` cache must be rebuilt (delete the `.tbin` file) when `computeTrace()` or param catalog changes
- `StreamTraining`/`StreamValidation` callbacks must not retain `[]TunerTrace` batch references (reused)
- **New eval parameter**: Add to `initTunerParams()`, `computeTrace()`, `PrintParams()`. Update "What's tuned" lists. Delete `.tbin` cache
