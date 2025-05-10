# File: core/metrics.py

from prometheus_client import Counter, Histogram, start_http_server


def start_metrics_server(port: int = 8000):
    """
    Spins up a Prometheus metrics endpoint at /metrics on the given port.
    """
    start_http_server(port)


# Prometheus metrics definitions
emails_fetched       = Counter("emails_fetched_total",       "Total emails fetched")
parse_success        = Counter("parse_success_total",        "Successful parser invocations")
parse_failure        = Counter("parse_failure_total",        "Failed parser invocations")
llm_latency          = Histogram("llm_latency_seconds",       "LLM parse latency in seconds")
regex_fallback_count = Counter("regex_fallback_total",        "Regex fallback invocations")
exports_done         = Counter("transactions_exported_total", "Number of transactions exported")