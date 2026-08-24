import numpy as np


def blocking_probability(rho: float, capacity: int) -> float:
    if rho <= 0.0:
        return 0.0
    if abs(rho - 1.0) < 1e-12:
        return capacity / (capacity + 1.0)

    with np.errstate(over="ignore"):
        inverse_k1 = np.exp(-(capacity + 1) * np.log(rho))
    if np.isinf(inverse_k1):
        return 0.0
    denominator = inverse_k1 - 1.0
    return float(((1.0 - rho) / rho) / denominator)


def expected_queue_occupation(rho: float, capacity: int) -> float:
    if rho <= 0.0:
        return 0.0
    if abs(rho - 1.0) < 1e-9:
        return capacity / 2.0

    with np.errstate(over="ignore"):
        inverse_k1 = np.exp(-(capacity + 1) * np.log(rho))
    if np.isinf(inverse_k1):
        return float(rho / (1.0 - rho))
    transient = (capacity + 1) / (inverse_k1 - 1.0)
    return float(rho / (1.0 - rho) - transient)


def expected_switch_delay(arrival_rate: float,
                          service_rate: float,
                          capacity: int) -> float:
    if arrival_rate <= 0.0 or service_rate <= 0.0:
        return 0.0
    rho = arrival_rate / service_rate
    p_b = blocking_probability(rho, capacity)
    queue_occupation = expected_queue_occupation(rho, capacity)
    effective_throughput = arrival_rate * max(1.0 - p_b, 1e-12)
    return float(queue_occupation / effective_throughput)


def expected_lost_rate(arrival_rate: float,
                       service_rate: float,
                       capacity: int) -> float:
    if arrival_rate <= 0.0:
        return 0.0
    rho = arrival_rate / service_rate
    p_b = blocking_probability(rho, capacity)
    return float(arrival_rate * p_b)


def network_queue_metrics(per_switch_arrival_rates: np.ndarray,
                          service_rate: float,
                          capacity: int) -> dict:
    rates = np.asarray(per_switch_arrival_rates, dtype=np.float64)
    delays = np.array([expected_switch_delay(lam, service_rate, capacity)
                       for lam in rates])
    lost = np.array([expected_lost_rate(lam, service_rate, capacity)
                     for lam in rates])

    total_offered = rates.sum()
    total_lost = lost.sum()

    if total_offered > 0:
        mean_delay = float((rates * delays).sum() / total_offered)
        loss_fraction = float(total_lost / total_offered)
    else:
        mean_delay = 0.0
        loss_fraction = 0.0

    return {
        "mean_delay": mean_delay,
        "loss_fraction": loss_fraction,
        "throughput": float(total_offered - total_lost),
        "per_switch_delays": delays,
        "per_switch_loss_rates": lost,
    }
