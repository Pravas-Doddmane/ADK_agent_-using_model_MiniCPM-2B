import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator_agent.classifier import classify_intent, require_hf_token

# Load test cases
with open("benchmark/test_cases.json", "r") as f:
    test_cases = json.load(f)

require_hf_token()

def classify(user_input):
    result = classify_intent(user_input)
    return result

# Warm-up (cold start)
print("Warming up model...")
classify("Hello")
print("Warm-up done.\n")

results = []
latencies = []
correct = 0
api_error_count = 0
fallback_count = 0
total_retries = 0
model_call_disabled_count = 0

for i, case in enumerate(test_cases, 1):
    inp = case["input"]
    expected = case["expected"]
    try:
        output = classify(inp)
        predicted = output.label
        lat = output.latency_s
        latencies.append(lat)
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        if output.api_error:
            if output.api_error == "MODEL_CALL_DISABLED":
                model_call_disabled_count += 1
            else:
                api_error_count += 1
        if output.used_fallback:
            fallback_count += 1
        total_retries += output.retries
        results.append({
            "input": inp,
            "expected": expected,
            "predicted": predicted,
            "latency_ms": round(lat * 1000, 2),
            "source": output.source,
            "retries": output.retries,
            "used_fallback": output.used_fallback,
            "api_error": output.api_error,
            "correct": is_correct
        })
        print(
            f"{i:2d}: {inp[:40]:40} -> {predicted:30} "
            f"({(lat*1000):.1f} ms, src={output.source}, retries={output.retries})"
        )
    except Exception as e:
        print(f"Error on '{inp}': {e}")
        api_error_count += 1
        results.append({
            "input": inp,
            "expected": expected,
            "predicted": "ERROR",
            "latency_ms": None,
            "source": "exception",
            "retries": None,
            "used_fallback": False,
            "api_error": str(e),
            "correct": False
        })

# Compute metrics
n = len(test_cases)
accuracy = (correct / n) * 100 if n > 0 else 0
avg_latency = (sum(latencies) / len(latencies)) * 1000 if latencies else 0
sorted_lats = sorted(latencies)
p95 = sorted_lats[int(0.95 * len(sorted_lats)) - 1] * 1000 if latencies else 0
total_time = sum(latencies)
throughput = n / total_time if total_time > 0 else 0

# Token estimate for this pipeline
prompt_tokens = 180
output_tokens = 8
total_tokens = (prompt_tokens + output_tokens) * n

summary = {
    "total_test_cases": n,
    "correct": correct,
    "accuracy_percent": round(accuracy, 2),
    "avg_latency_ms": round(avg_latency, 2),
    "p95_latency_ms": round(p95, 2),
    "throughput_rps": round(throughput, 2),
    "api_error_count": api_error_count,
    "model_call_disabled_count": model_call_disabled_count,
    "fallback_count": fallback_count,
    "total_retries": total_retries,
    "estimated_total_tokens": total_tokens,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

output = {
    "summary": summary,
    "results": results
}

with open("benchmark/results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n--- Benchmark Summary ---")
for k, v in summary.items():
    print(f"{k}: {v}")
print("\nResults saved to benchmark/results.json")
