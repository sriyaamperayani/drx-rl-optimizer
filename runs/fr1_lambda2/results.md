# Results

Held-out evaluation: 200 episodes x 60 windows of 1000 ms, beam search 0 ms, lambda = 2. Training: 5000 episodes (107 s).

| Policy | Return | Energy saving | Packets over budget | Mean latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| operator_default | -29.04 | 61.3% | 5.69% | 30.0 ms | 132.7 ms |
| best_static | -22.88 | 63.9% | 0.51% | 14.8 ms | 33.4 ms |
| heuristic | -22.03 | 65.8% | 0.82% | 17.1 ms | 62.8 ms |
| rl_q_learning | -20.26 | 71.3% | 3.12% | 23.2 ms | 68.8 ms |

best_static = `inact=10ms/cycle=40ms` (chosen with hindsight on the evaluation traces).
