import numpy as np


def delay_reward(mean_end_to_end_delay: float, worst_path_drain_time: float) -> float:
    if worst_path_drain_time <= 0.0:
        return 1.0
    value = 1.0 - mean_end_to_end_delay / worst_path_drain_time
    return float(np.clip(value, 0.0, 1.0))


def loss_reward(total_lost_rate: float, total_offered_rate: float) -> float:
    if total_offered_rate <= 0.0:
        return 1.0
    value = 1.0 - total_lost_rate / total_offered_rate
    return float(np.clip(value, 0.0, 1.0))


def combined_reward(delay_r: float, loss_r: float, alpha: float) -> float:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    value = alpha * delay_r + (1.0 - alpha) * loss_r
    return float(np.clip(value, 0.0, 1.0))
