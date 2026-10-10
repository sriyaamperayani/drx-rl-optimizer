# Results

Held-out evaluation: 200 episodes x 60 windows of 1000 ms, beam search 4 ms, lambda = 5. Training: 5000 episodes (110 s).

| Policy | Return | Energy saving | Packets over budget | Mean latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| operator_default | -38.32 | 58.9% | 5.84% | 30.2 ms | 132.6 ms |
| best_static | -32.21 | 53.5% | 1.32% | 25.9 ms | 131.2 ms |
| heuristic | -29.08 | 56.8% | 0.92% | 18.2 ms | 61.8 ms |
| rl_q_learning | -29.77 | 54.9% | 0.79% | 27.6 ms | 123.5 ms |

best_static = `inact=200ms/cycle=160ms` (chosen with hindsight on the evaluation traces).
