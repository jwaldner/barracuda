# Configuration Reference — `config.yaml`

This document explains the keys found in `config.yaml` and how to safely configure the application.

> Note: Never commit live API secrets to a public repository. Use placeholders, environment variables, or a secure secrets manager.

## Overview

The `config.yaml` file contains configuration for the Alpaca API, trading/engine defaults, and S&P 500 symbol update settings.

Sections:
- `alpaca` — API keys and credentials
- `trading` — trading defaults used by analyses
- `engine` — engine runtime / performance options
- `sp500` — S&P 500 symbol update settings

## Key reference

### `alpaca`
- `api_key`: string — Alpaca API key ID. **Do not** store production secrets in the repo.
- `secret_key`: string — Alpaca secret key.

Recommended alternatives:
- Replace values with placeholders in repo and set real keys via environment variables (preferred).
- Example placeholder in YAML: `api_key: "<ALPACA_API_KEY>"`

Example (safe) environment usage pattern (application-level):
- Use environment variables `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` and load them at startup.

YAML example (placeholder):
```yaml
alpaca:
  api_key: "<ALPACA_API_KEY>"
  secret_key: "<ALPACA_SECRET_KEY>"
```

Security recommendations:
- Add `config.yaml` to `.gitignore` if it contains secrets.
- Use `config.yaml.template` in the repo with placeholders, and keep the real `config.yaml` out of version control.
- Consider using a secrets manager (vault, AWS Secrets Manager) for production.

### `trading`
- `default_cash`: number — default available cash used for analysis (e.g. `70000`).
- `target_delta`: number — target delta value for puts (e.g. `0.25`).
- `max_results`: integer — number of top results to return (e.g. `25`).
- `default_stocks`: list — an optional list of stock symbols to use instead of the S&P 500 ranking. Example: `default_stocks: ["AAPL","MSFT"]`.

### `engine`
- `execution_mode`: string — execution mode (common values: `auto`, `manual`). Controls whether the engine runs automatically or waits for manual triggers.
- `batch_size`: integer — number of items processed per batch (e.g. `1000`).
- `enable_benchmarks`: boolean — enable benchmark logging and measurements.
- `workload_factor`: number — multiplier to scale computational workload (1.0 = normal).

Notes:
- Increasing `batch_size` may improve throughput but increase memory usage.
- `workload_factor` can be used to simulate heavier computation for testing.

### `sp500`
- `auto_update`: boolean — whether to auto-update the S&P 500 symbol list at startup.
- `update_interval_hours`: integer — hours between updates (e.g. `168` for weekly updates).
- `data_source`: string — where to get S&P 500 symbol data (e.g. `github_csv`).

How the auto-update works
- When `sp500.auto_update` is `true`, the application will attempt to refresh the S&P 500 symbol list at startup and then again on the interval set by `sp500.update_interval_hours`.
- The data is fetched from the `sp500.data_source` (for example `github_csv`), parsed, and used to update the local symbols store used by the app.

Turn it off (recommended for most users)
- To disable automatic updates, set `sp500.auto_update: false` in your `config.yaml`. This prevents the app from fetching/updating symbols on startup.
- Example (safe `config.yaml`):

```yaml
sp500:
  auto_update: false
  update_interval_hours: 168
  data_source: "github_csv"
```

Make the change persistent for new clones
- Update `config.yaml.template` to include `auto_update: false` so new local copies default to disabled.
- Keep the live `config.yaml` with real secrets and local preferences out of version control (add to `.gitignore`).

## Sample safe `config.yaml.template`

Include this file in the repo and copy to `config.yaml` locally before filling keys.

```yaml
# config.yaml.template — keep placeholders, do not store real secrets
alpaca:
  api_key: "<ALPACA_API_KEY>"
  secret_key: "<ALPACA_SECRET_KEY>"

trading:
  default_cash: 70000
  target_delta: 0.25
  max_results: 25
  default_stocks: []

engine:
  execution_mode: "auto"
  batch_size: 1000
  enable_benchmarks: true
  workload_factor: 1.0

sp500:
  auto_update: true
  update_interval_hours: 168
  data_source: "github_csv"
```

## Quick workflows

- Create a local `config.yaml` from the template:

```bash
cp config.yaml.template config.yaml
# Edit config.yaml to add your real secrets locally (do not commit)
```

- Use environment variables instead (if app supports it):

```bash
export ALPACA_API_KEY=your_key_here
export ALPACA_SECRET_KEY=your_secret_here
# Run the app which reads these env vars
```

- Add `config.yaml` to `.gitignore`:

```
# Ignore local config with secrets
/config.yaml
```

## Notes and next steps
- There is a `config.yaml.template` in the repository; prefer committing that with placeholders instead of real keys.
- If you want, I can update `config.yaml.template` to replace any real keys with placeholders and/or modify the application to prefer environment variables.

If you'd like, I can now:
- Update `config.yaml.template` to remove secrets and insert placeholders.
- Modify code to load `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` from environment variables where appropriate.
- Create a short contributor note in `README.md` about handling secrets.

