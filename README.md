# Agent Spectrum Operating System

A proof-of-concept OS where agents are scheduled, routed, and composed using **conservation spectral analysis**.

## Core Idea

Each agent IS its spectral fingerprint (not its API description). The system uses graph Laplacian eigenanalysis to:

- **Register** agents by their spectral fingerprint (eigenvalues of capability coupling graph)
- **Schedule** tasks using Fiedler vector partitioning of task-agent compatibility graphs
- **Compose** agents by merging Laplacians and checking conservation ratios
- **Detect anomalies** by observing conservation drops when agents degrade

## Architecture

```
ConservationAgent          → Each agent = capability graph → Laplacian → spectral fingerprint
AgentSpectrumOS.register() → Fingerprint the agent
AgentSpectrumOS.schedule() → Fiedler routing on compatibility graph
AgentSpectrumOS.compose()  → Merge Laplacians, check conservation before deploy
detect_anomaly()           → Inject fault, measure spectral gap drop
```

## Run

```bash
python3 spectrum_os.py
```

## Scenarios

| # | Scenario | Expected | Result |
|---|----------|----------|--------|
| 1 | Single task → Analyst | ✅ Success | Conservation 1.0 |
| 2 | Builder + Validator | ✅ Approved | High alignment (0.67) |
| 3 | Analyst + Operator | ❌ Rejected | Low alignment (0.14) |
| 4 | Full research team | ✅ Approved | Coherent pipeline (0.69) |
| 5 | Inject failing Builder | 🔴 Anomaly detected | Spectral gap drops 99% |

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
