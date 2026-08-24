import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_metrics(metrics_path: str) -> Dict:
    with open(metrics_path) as f:
        return json.load(f)


def extract_learning_curves(data: Dict) -> Dict[str, List[float]]:
    epochs = []
    agent = []
    for entry in data["history"]:
        if "test_agent_ratio_mean" in entry:
            epochs.append(entry["epoch"])
            agent.append(entry["test_agent_ratio_mean"])
    baselines = {}
    if "test_prev_ratio_mean" in data["history"][-1]:
        final = data["history"][-1]
        baselines = {
            "Prev": (final["test_prev_ratio_mean"], final.get("test_prev_ratio_std", 0.0)),
            "Avg_k": (final["test_avgk_ratio_mean"], final.get("test_avgk_ratio_std", 0.0)),
            "Oblivious": (final["test_oblivious_ratio_mean"], final.get("test_oblivious_ratio_std", 0.0)),
        }
    return {"epochs": epochs, "agent": agent, "baselines": baselines}


def summarize_run(data: Dict) -> Dict:
    curves = extract_learning_curves(data)
    final = data["history"][-1]
    first_agent = next((e["test_agent_ratio_mean"]
                        for e in data["history"]
                        if "test_agent_ratio_mean" in e), None)
    return {
        "config": data["config"],
        "initial_agent_ratio": first_agent,
        "final_agent_ratio": final.get("final_agent_ratio_mean", curves["agent"][-1] if curves["agent"] else None),
        "prev_ratio": final.get("final_prev_ratio_mean"),
        "avgk_ratio": final.get("final_avgk_ratio_mean"),
        "oblivious_ratio": final.get("final_oblivious_ratio_mean"),
        "agent_delivered_min": final.get("final_agent_delivered_min"),
    }
