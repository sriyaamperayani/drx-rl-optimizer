# Results

Held-out evaluation: 200 episodes x 60 windows of 1000 ms, beam search 4 ms, lambda = 2. Training: 5000 episodes (108 s).

| Policy | Return | Energy saving | Packets over budget | Mean latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| operator_default | -30.66 | 58.9% | 5.84% | 30.2 ms | 132.6 ms |
| best_static | -30.24 | 53.5% | 1.32% | 25.9 ms | 131.2 ms |
| heuristic | -27.75 | 56.8% | 0.92% | 18.2 ms | 61.8 ms |
| rl_q_learning | -23.82 | 68.6% | 5.14% | 45.6 ms | 189.7 ms |

best_static = `inact=200ms/cycle=160ms` (chosen with hindsight on the evaluation traces).
