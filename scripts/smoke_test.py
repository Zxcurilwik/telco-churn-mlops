"""Smoke test against a running API (local container, Space or Render URL).

python scripts/smoke_test.py http://localhost:8000
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def call(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def wait_until_healthy(base: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return call(f"{base}/health")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            if time.monotonic() > deadline:
                raise SystemExit(f"API at {base} not healthy after {timeout:.0f}s: {exc}") from exc
            time.sleep(1)


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 60

    health = wait_until_healthy(base, timeout)
    assert health["status"] == "ok", health
    print(f"health      ok  model_version={health['model_version']}")

    single = call(f"{base}/predict", json.loads((EXAMPLES / "customer.json").read_text()))
    assert 0 <= single["churn_probability"] <= 1, single
    print(f"predict     ok  p={single['churn_probability']} risk={single['risk_level']}")

    batch = call(f"{base}/predict/batch", json.loads((EXAMPLES / "batch.json").read_text()))
    assert len(batch["predictions"]) == 2, batch
    print(f"batch       ok  p={[p['churn_probability'] for p in batch['predictions']]}")

    info = call(f"{base}/model/info")
    print(f"model/info  ok  {info['candidate']}  test_roc_auc={info['metrics_test']['roc_auc']}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
