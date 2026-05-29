# agent-spectrum-os

A proof-of-concept operating system where agents are scheduled, routed, and composed using spectral graph theory. Each agent IS its Laplacian — eigenvalues are identity, the Fiedler vector is routing, and conservation ratios are confidence.

## What This Gives You

- **Spectral fingerprinting**: register agents by the eigenvalues of their capability coupling graph
- **Fiedler scheduling**: route tasks to agents using Fiedler vector partitioning of task-agent compatibility
- **Laplacian composition**: merge agents by combining their Laplacians, check conservation before deploying
- **Anomaly detection**: inject faults and detect degradation by observing spectral gap drops

## Quick Start

```bash
pip install numpy
python3 spectrum_os.py
```

This runs 5 scenarios:

| # | Scenario | Outcome |
|---|----------|---------|
| 1 | Single task → Analyst agent | ✅ Routed correctly, conservation 1.0 |
| 2 | Builder + Validator composition | ✅ Approved, alignment 0.67 |
| 3 | Analyst + Operator composition | ❌ Rejected, alignment 0.14 |
| 4 | Full research team (4 agents) | ✅ Approved, coherence 0.69 |
| 5 | Inject failing Builder | 🔴 Anomaly detected, spectral gap drops 99% |

## Architecture

```
ConservationAgent          → capability graph → Laplacian → spectral fingerprint
AgentSpectrumOS.register() → fingerprint the agent
AgentSpectrumOS.schedule() → Fiedler routing on compatibility graph
AgentSpectrumOS.compose()  → merge Laplacians, verify conservation
detect_anomaly()           → inject fault, measure spectral gap change
```

### Key Concepts

| Concept | Role |
|---------|------|
| Laplacian | Agent's full capability structure as a spectral object |
| Eigenvalues | Spectral fingerprint — identifies agent "shape" |
| Conservation ratio | Confidence — how well-structured the agent's state is |
| Fiedler vector | Routing — which agents should handle which tasks |
| Spectral alignment | Compatibility — cosine similarity of eigenvalue spectra |

## How It Fits

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem. This is the experimental research layer on top of:

- **agent-manifest-rs** — manifests provide the capability graphs that become Laplacians
- **agent-native-language** — agents communicating purely through spectral messages
- **agent-handshake-rs** — handshake protocol for establishing spectral communication channels

## Testing

All 5 scenarios run as integration tests via `python3 spectrum_os.py`. Each scenario asserts expected outcomes (success/rejection/anomaly detection).

## Installation

```bash
pip install numpy
```

Python 3 with NumPy. No other dependencies.
