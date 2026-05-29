# agent-spectrum-os — Spectral Agent Operating System

**Agents are Laplacians. Eigenvalues are identity. The Fiedler vector is routing. Conservation ratios are confidence.**

## What This Gives You

- **Spectral agents** — each agent *is* its Laplacian matrix, derived from capability coupling
- **Spectral fingerprinting** — eigenvalues identify agents, spectral gap measures resilience
- **Fiedler routing** — use the Fiedler vector for optimal task-to-agent routing
- **Conservation ratios** — agent confidence derived from energy conservation in the spectral domain
- **Spectral composition** — merge agents by merging their graphs, compose capabilities algebraically

## Quick Start

```bash
pip install agent-spectrum-os
```

```python
from spectrum_os import ConservationAgent, SpectrumScheduler

# Create spectral agents
agent_a = ConservationAgent(
    name="rust-builder",
    capabilities={"compilation": 0.9, "testing": 0.8, "docs": 0.3},
    connections={"python-builder": 0.7},
)

agent_b = ConservationAgent(
    name="python-builder",
    capabilities={"compilation": 0.6, "testing": 0.9, "docs": 0.8},
    connections={"rust-builder": 0.7},
)

# Spectral fingerprint
fp = agent_a.spectral_fingerprint
print(f"Eigenvalues: {fp['eigenvalues']}")
print(f"Spectral gap: {fp['spectral_gap']:.3f}")
print(f"Fiedler value: {fp['fiedler_value']:.3f}")

# Schedule tasks using spectral partitioning
scheduler = SpectrumScheduler(agents=[agent_a, agent_b])
assignment = scheduler.assign(task="run benchmarks")
print(f"Assigned to: {assignment.agent}")
print(f"Confidence: {assignment.confidence:.2f}")

# Compose agents
composed = agent_a.compose(agent_b)
print(f"Combined capabilities: {composed.capabilities}")
```

## API Reference

### `ConservationAgent(name, capabilities, connections)`
- `spectral_fingerprint` → eigenvalues, spectral_gap, fiedler_value
- `compose(other)` → merged agent
- `confidence()` → conservation ratio

### `SpectrumScheduler(agents)`
- `assign(task)` → spectral routing result
- `partition(n_groups)` → Fiedler-based grouping

## How It Fits

A proof-of-concept from the [SuperInstance fleet](https://github.com/SuperInstance) exploring whether spectral graph theory can provide a principled foundation for agent identity and routing.

- **[agent-grid](https://github.com/SuperInstance/agent-grid)** — Practical grid topology (this is the mathematical exploration)
- **[captain](https://github.com/SuperInstance/captain)** — Fleet coordination
- **[cluster-orchestrator](https://github.com/SuperInstance/cluster-orchestrator)** — Cluster scheduling

## Testing

```bash
python -m pytest tests/
```

## Installation

```bash
pip install agent-spectrum-os
```

Python 3.10+. Requires NumPy. MIT license.
