N = 16
tau = 5
denom = 2

sparsity_thresholds = {
    "easy": N / (N**2 * tau),  # for N = 4, tau = 5, prob = 4 / 80 = 0.05
    "med_easy": 2 * N / (N**2 * tau),  # for N = 4, tau = 5, prob = 8 / 80 = 0.1
    "med_hard": 3 * N / (N**2 * tau),  # for N = 4, tau = 5, prob = 12 / 80 = 0.15
    "hard": (N + N * (N - 1) / denom) / (N**2 * tau),  # for N = 4, tau = 5, prob = 10 / 80 = 0.125
}

for difficulty in sparsity_thresholds.keys():
    print(f"difficulty: {difficulty}, sparsity_threshold: {sparsity_thresholds[difficulty]}")
