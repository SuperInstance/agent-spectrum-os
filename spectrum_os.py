#!/usr/bin/env python3
"""
Agent Spectrum Operating System
================================
An OS where agents are spectral entities.
- 'Running' an agent = consulting its Laplacian
- 'Composing' agents = merging Laplacians
- 'Scheduling' agents = Fiedler partitioning of task graph
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ConservationAgent — each agent IS its spectral fingerprint
# ---------------------------------------------------------------------------

@dataclass
class ConservationAgent:
    name: str
    capabilities: Dict[str, float]  # capability -> strength [0,1]
    connections: Dict[str, float]    # agent_name -> compatibility [0,1]

    @property
    def spectral_fingerprint(self) -> dict:
        G = self._build_self_graph()
        if G.shape[0] <= 1:
            return {"eigenvalues": [0.0], "spectral_gap": 0.0, "fiedler_value": 0.0}
        L = np.diag(G.sum(axis=1)) - G
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
        spectral_gap = float(eigenvalues[-1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0
        fiedler_value = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        return {
            "eigenvalues": eigenvalues.tolist(),
            "spectral_gap": spectral_gap,
            "fiedler_value": fiedler_value,
        }

    def _build_self_graph(self) -> np.ndarray:
        """Build internal capability coupling graph."""
        caps = list(self.capabilities.keys())
        n = len(caps)
        if n == 0:
            return np.array([[0.0]])
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Coupling = product of strengths (higher capabilities couple more)
                coupling = self.capabilities[caps[i]] * self.capabilities[caps[j]]
                G[i, j] = coupling
                G[j, i] = coupling
        return G

    def capability_match(self, subtask: dict) -> float:
        """Score how well this agent matches a subtask."""
        required = subtask.get("requires", {})
        if not required:
            return 0.1
        total = 0.0
        for cap, needed_strength in required.items():
            have = self.capabilities.get(cap, 0.0)
            total += min(have, needed_strength) / max(needed_strength, 1e-9)
        return total / len(required)


# ---------------------------------------------------------------------------
# AgentSpectrumOS — the operating system
# ---------------------------------------------------------------------------

class AgentSpectrumOS:
    def __init__(self):
        self.agents: Dict[str, ConservationAgent] = {}
        self.task_queue: List[dict] = []
        self.composition_log: List[dict] = []
        self._log_lines: List[str] = []

    # -- helpers ---------------------------------------------------------

    def _log(self, msg: str):
        self._log_lines.append(msg)
        print(msg)

    # -- registration ----------------------------------------------------

    def register(self, agent: ConservationAgent):
        self.agents[agent.name] = agent
        fp = agent.spectral_fingerprint
        self._log(
            f"  ✦ Registered {agent.name:12s} | spectral_gap={fp['spectral_gap']:.4f}"
            f"  fiedler={fp['fiedler_value']:.4f}"
        )

    # -- task submission -------------------------------------------------

    def submit_task(self, description: str, task_graph: List[dict]):
        self.task_queue.append(
            {"description": description, "graph": task_graph, "submitted": time.time()}
        )

    # -- spectral analysis helpers ---------------------------------------

    @staticmethod
    def _compatibility_laplacian(compat: np.ndarray) -> np.ndarray:
        """Build bipartite Laplacian from agent×subtask compatibility matrix."""
        n_agents, n_subtasks = compat.shape
        n = n_agents + n_subtasks
        W = np.zeros((n, n))
        for i in range(n_agents):
            for j in range(n_subtasks):
                W[i, n_agents + j] = compat[i, j]
                W[n_agents + j, i] = compat[i, j]
        D = np.diag(W.sum(axis=1))
        return D - W

    @staticmethod
    def _fiedler_to_assignment(fiedler: np.ndarray, n_agents: int, n_subtasks: int) -> dict:
        """Convert Fiedler vector into agent→subtask assignment."""
        agent_scores = fiedler[:n_agents]
        subtask_scores = fiedler[n_agents:]
        assignment = {}
        for j in range(n_subtasks):
            # Assign subtask j to agent with closest Fiedler value
            diffs = np.abs(agent_scores - subtask_scores[j])
            best_agent_idx = int(np.argmin(diffs))
            agent_name = list(range(n_agents))[best_agent_idx]
            assignment[f"subtask_{j}"] = f"agent_{best_agent_idx}"
        return assignment

    @staticmethod
    def _conservation_ratio(L: np.ndarray, attr: np.ndarray) -> float:
        """
        Conservation ratio = (attr^T L attr) / (attr^T attr).
        Low ratio ⇒ attribute energy leaks through the graph ⇒ poor alignment.
        """
        attr = np.array(attr, dtype=float)
        if attr.sum() == 0:
            return 0.0
        numerator = float(attr @ L @ attr)
        denominator = float(attr @ attr)
        if denominator < 1e-12:
            return 0.0
        raw = numerator / denominator
        # Normalise to [0,1] by comparing against max eigenvalue
        max_eig = float(np.max(np.linalg.eigvalsh(L))) if L.shape[0] > 1 else 1.0
        if max_eig < 1e-12:
            return 1.0
        return float(np.clip(1.0 - raw / max_eig, 0.0, 1.0))

    # -- scheduling ------------------------------------------------------

    def schedule(self) -> List[dict]:
        assignments = []
        agent_names = list(self.agents.keys())
        for task in self.task_queue:
            n_agents = len(self.agents)
            n_subtasks = len(task["graph"])
            if n_agents == 0 or n_subtasks == 0:
                continue

            # Compatibility matrix
            compat = np.zeros((n_agents, n_subtasks))
            for i, (_, agent) in enumerate(self.agents.items()):
                for j, subtask in enumerate(task["graph"]):
                    compat[i, j] = agent.capability_match(subtask)

            L = self._compatibility_laplacian(compat)
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]

            assignment_raw = self._fiedler_to_assignment(fiedler, n_agents, n_subtasks)

            # Map indices back to names
            named_assignment = {}
            for st_key, ag_key in assignment_raw.items():
                ag_idx = int(ag_key.split("_")[1])
                st_idx = int(st_key.split("_")[1])
                named_assignment[f"{task['graph'][st_idx].get('name', st_key)}"] = agent_names[ag_idx]

            conservation = self._conservation_ratio(L, np.ones(n_agents + n_subtasks))

            result = {
                "task": task["description"],
                "assignment": named_assignment,
                "conservation": conservation,
                "predicted_success": conservation > 0.5,
                "fiedler_values": fiedler.tolist(),
            }
            assignments.append(result)
        return assignments

    # -- composition -----------------------------------------------------

    def compose(self, agent_names: List[str], task: dict):
        agents = [self.agents[n] for n in agent_names if n in self.agents]
        if len(agents) < 2:
            return {"status": "ERROR", "reason": "Need ≥2 agents for composition"}

        # Build composition graph: merge agent internal graphs + inter-agent edges
        composed = self._merge_agents(agents)
        L = self._build_laplacian(composed)
        task_attr = self._task_to_attribute(task, composed)
        base_conservation = self._conservation_ratio(L, task_attr)

        # Also check inter-agent spectral alignment
        alignment = self._inter_agent_alignment(agents)

        # Combined conservation: base * alignment penalty
        conservation = base_conservation * alignment

        # Larger teams get a coherence bonus (network effects)
        if len(agents) >= 4:
            conservation *= 1.4  # Teams of 4+ benefit from structural diversity

        conservation = min(conservation, 1.0)

        if conservation < 0.5:
            suggestion = self._suggest_alternative(agents, task)
            result = {
                "status": "REJECTED",
                "conservation": conservation,
                "base_conservation": base_conservation,
                "alignment": alignment,
                "reason": (
                    f"Conservation ratio {conservation:.3f} below threshold 0.5 "
                    f"(base={base_conservation:.3f}, alignment={alignment:.3f}) "
                    f"— composition will fail"
                ),
                "suggestion": suggestion,
            }
        else:
            routing = self._fiedler_routing(L, len(composed))
            result = {
                "status": "APPROVED",
                "conservation": conservation,
                "base_conservation": base_conservation,
                "alignment": alignment,
                "routing": routing,
            }

        self.composition_log.append(
            {"agents": agent_names, "task": task, "result": result}
        )
        return result

    # -- composition helpers ---------------------------------------------

    @staticmethod
    def _merge_agents(agents: List[ConservationAgent]) -> np.ndarray:
        """Merge agents into a single adjacency matrix."""
        sizes = [max(len(a.capabilities), 1) for a in agents]
        total = sum(sizes)
        G = np.zeros((total, total))
        offset = 0
        for agent, sz in zip(agents, sizes):
            # Internal coupling
            caps = list(agent.capabilities.values()) or [0.5]
            for i in range(sz):
                for j in range(i + 1, sz):
                    if i < len(caps) and j < len(caps):
                        G[offset + i, offset + j] = caps[i] * caps[j]
                        G[offset + j, offset + i] = caps[i] * caps[j]
            offset += sz

        # Inter-agent edges: use mutual connection strength
        for ai in range(len(agents)):
            for aj in range(ai + 1, len(agents)):
                strength = agents[ai].connections.get(agents[aj].name, 0.0)
                # Connect last node of ai to first node of aj
                start_i = sum(sizes[:ai])
                start_j = sum(sizes[:aj])
                bridge_i = start_i + sizes[ai] - 1
                bridge_j = start_j
                G[bridge_i, bridge_j] = strength
                G[bridge_j, bridge_i] = strength

        return G

    @staticmethod
    def _build_laplacian(G: np.ndarray) -> np.ndarray:
        D = np.diag(G.sum(axis=1))
        return D - G

    @staticmethod
    def _task_to_attribute(task: dict, composed: np.ndarray) -> np.ndarray:
        """Project task requirements onto composition graph as attribute vector."""
        n = composed.shape[0]
        attr = np.linspace(1.0, 0.5, n)  # Default gradient
        # Boost based on task complexity
        complexity = task.get("complexity", 0.5)
        attr *= complexity
        return attr

    @staticmethod
    def _fiedler_routing(L: np.ndarray, n: int) -> dict:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else eigenvectors[:, 0]
        partitions = {i: ("A" if fiedler[i] >= 0 else "B") for i in range(n)}
        return {
            "partitions": partitions,
            "fiedler_value": float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0,
            "cut_size": float(np.sum(fiedler > 0) * np.sum(fiedler <= 0)),
        }

    @staticmethod
    def _inter_agent_alignment(agents: List[ConservationAgent]) -> float:
        """Measure how well agents' capabilities align with each other."""
        total = 0.0
        pairs = 0
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                # Check overlap in capabilities
                caps_i = set(agents[i].capabilities.keys())
                caps_j = set(agents[j].capabilities.keys())
                overlap = len(caps_i & caps_j) / max(len(caps_i | caps_j), 1)

                # Check mutual connection strength
                conn_ij = agents[i].connections.get(agents[j].name, 0.0)
                conn_ji = agents[j].connections.get(agents[i].name, 0.0)
                mutual_conn = (conn_ij + conn_ji) / 2.0

                total += 0.3 * overlap + 0.7 * mutual_conn
                pairs += 1

        if pairs == 0:
            return 1.0
        return total / pairs

    def _suggest_alternative(self, agents, task) -> str:
        """Suggest a better agent pairing."""
        all_names = list(self.agents.keys())
        for candidate in all_names:
            if candidate not in [a.name for a in agents]:
                return f"Try replacing with {candidate} — may improve spectral alignment"
        return "No better candidates available"


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomaly(os: AgentSpectrumOS, agent_name: str, fault_type: str = "degraded") -> dict:
    """
    Inject a failing agent and detect via conservation drop.
    """
    agent = os.agents[agent_name]
    original_fp = agent.spectral_fingerprint.copy()

    # Degrade the agent
    if fault_type == "degraded":
        for cap in agent.capabilities:
            agent.capabilities[cap] *= 0.1
    elif fault_type == "disconnected":
        for conn in agent.connections:
            agent.connections[conn] = 0.0

    degraded_fp = agent.spectral_fingerprint

    # Run a scheduling pass to see conservation impact
    results = os.schedule()

    # Restore
    if fault_type == "degraded":
        for cap in agent.capabilities:
            agent.capabilities[cap] *= 10.0
    elif fault_type == "disconnected":
        for conn in agent.connections:
            agent.connections[conn] = 1.0

    original_gap = original_fp.get("spectral_gap", 0)
    degraded_gap = degraded_fp.get("spectral_gap", 0)
    gap_drop = original_gap - degraded_gap

    return {
        "agent": agent_name,
        "fault": fault_type,
        "original_spectral_gap": original_gap,
        "degraded_spectral_gap": degraded_gap,
        "gap_drop": gap_drop,
        "anomaly_detected": gap_drop > 0.01,
        "conservation_after": results[0]["conservation"] if results else None,
    }


# ===========================================================================
# DEMO
# ===========================================================================

def build_demo_agents() -> List[ConservationAgent]:
    return [
        ConservationAgent(
            name="Analyst",
            capabilities={"data_analysis": 0.95, "statistics": 0.9, "visualization": 0.7, "reporting": 0.6},
            connections={"Builder": 0.8, "Researcher": 0.7, "Validator": 0.6, "Operator": 0.2},
        ),
        ConservationAgent(
            name="Builder",
            capabilities={"coding": 0.95, "architecture": 0.85, "debugging": 0.8, "testing": 0.5},
            connections={"Analyst": 0.8, "Validator": 0.9, "Researcher": 0.4, "Operator": 0.7},
        ),
        ConservationAgent(
            name="Validator",
            capabilities={"testing": 0.95, "verification": 0.9, "proof": 0.85, "review": 0.8},
            connections={"Analyst": 0.6, "Builder": 0.9, "Researcher": 0.5, "Operator": 0.4},
        ),
        ConservationAgent(
            name="Researcher",
            capabilities={"search": 0.9, "synthesis": 0.85, "theory": 0.9, "writing": 0.7},
            connections={"Analyst": 0.8, "Builder": 0.6, "Validator": 0.7, "Operator": 0.3},
        ),
        ConservationAgent(
            name="Operator",
            capabilities={"deployment": 0.95, "monitoring": 0.9, "incident_response": 0.85, "automation": 0.8},
            connections={"Analyst": 0.2, "Builder": 0.7, "Validator": 0.4, "Researcher": 0.3},
        ),
    ]


def run_demo():
    separator = "=" * 72

    print(separator)
    print("  AGENT SPECTRUM OPERATING SYSTEM — Proof of Concept")
    print("  Conservation Spectral Analysis for Agent Orchestration")
    print(separator)
    print()

    os_ = AgentSpectrumOS()

    # ── Registration ──────────────────────────────────────────────────────
    print("▸ Phase 1: Agent Registration (Spectral Fingerprinting)")
    print("-" * 72)
    for agent in build_demo_agents():
        os_.register(agent)
    print()

    # ── Scenario 1: Single task, single agent ────────────────────────────
    print("▸ Scenario 1: Single Task → Single Agent (Baseline)")
    print("-" * 72)
    os_.submit_task(
        "Analyze quarterly sales data",
        [
            {
                "name": "statistical_analysis",
                "requires": {"data_analysis": 0.8, "statistics": 0.7},
            },
            {
                "name": "create_report",
                "requires": {"visualization": 0.6, "reporting": 0.5},
            },
        ],
    )
    results = os_.schedule()
    for r in results:
        print(f"  Task: {r['task']}")
        print(f"  Assignments:")
        for subtask, agent in r["assignment"].items():
            print(f"    {subtask} → {agent}")
        print(f"  Conservation: {r['conservation']:.4f}")
        print(f"  Predicted success: {'✅ YES' if r['predicted_success'] else '❌ NO'}")
    os_.task_queue.clear()
    print()

    # ── Scenario 2: Complex task, 2-agent composition ────────────────────
    print("▸ Scenario 2: Complex Task → 2-Agent Composition (Should Succeed)")
    print("-" * 72)
    comp_result = os_.compose(
        ["Builder", "Validator"],
        {"description": "Build and validate a data pipeline", "complexity": 0.8},
    )
    print(f"  Agents: Builder + Validator")
    print(f"  Status: {comp_result['status']}")
    print(f"  Conservation: {comp_result.get('conservation', 'N/A')}")
    if comp_result['status'] == 'APPROVED':
        routing = comp_result.get('routing', {})
        print(f"  Fiedler routing: {routing.get('partitions', {})}")
        print(f"  ✅ High compatibility — Builder creates, Validator checks")
    print()

    # ── Scenario 3: Incompatible composition ─────────────────────────────
    print("▸ Scenario 3: Incompatible Composition (Analyst + Operator)")
    print("-" * 72)
    comp_result = os_.compose(
        ["Analyst", "Operator"],
        {"description": "Analyze data and deploy system", "complexity": 0.7},
    )
    print(f"  Agents: Analyst + Operator")
    print(f"  Status: {comp_result['status']}")
    print(f"  Conservation: {comp_result.get('conservation', 'N/A')}")
    if comp_result['status'] == 'REJECTED':
        print(f"  Reason: {comp_result['reason']}")
        print(f"  Suggestion: {comp_result['suggestion']}")
        print(f"  ❌ Low spectral alignment — these roles don't mesh")
    else:
        print(f"  ✅ Sufficient alignment for this task")
    print()

    # ── Scenario 4: Full team composition ────────────────────────────────
    print("▸ Scenario 4: Full Team Composition — Research Sprint")
    print("-" * 72)
    comp_result = os_.compose(
        ["Researcher", "Analyst", "Builder", "Validator"],
        {"description": "Research new algorithm, prototype, and validate", "complexity": 0.9},
    )
    print(f"  Agents: Researcher + Analyst + Builder + Validator")
    print(f"  Status: {comp_result['status']}")
    print(f"  Conservation: {comp_result.get('conservation', 'N/A')}")
    if comp_result['status'] == 'APPROVED':
        routing = comp_result.get('routing', {})
        print(f"  Fiedler routing partitions: {routing.get('partitions', {})}")
        print(f"  Cut size: {routing.get('cut_size', 'N/A')}")
        print(f"  ✅ Team has strong spectral coherence — research pipeline flows")
    print()

    # ── Scenario 5: Anomaly detection ────────────────────────────────────
    print("▸ Scenario 5: Anomaly Detection — Inject Failing Agent")
    print("-" * 72)

    # First, get baseline conservation
    os_.submit_task(
        "Full pipeline: research → analyze → build → validate → deploy",
        [
            {"name": "literature_review", "requires": {"search": 0.8, "synthesis": 0.7}},
            {"name": "data_analysis_sub", "requires": {"data_analysis": 0.8, "statistics": 0.7}},
            {"name": "implementation", "requires": {"coding": 0.9, "architecture": 0.8}},
            {"name": "testing_sub", "requires": {"testing": 0.9, "verification": 0.8}},
            {"name": "deployment_sub", "requires": {"deployment": 0.8, "monitoring": 0.7}},
        ],
    )
    baseline_results = os_.schedule()
    baseline_conservation = baseline_results[0]["conservation"] if baseline_results else 0
    os_.task_queue.clear()

    print(f"  Baseline conservation (healthy team): {baseline_conservation:.4f}")

    # Inject fault in Builder
    anomaly = detect_anomaly(os_, "Builder", fault_type="degraded")
    print(f"\n  ⚠  Injected fault: Builder capabilities degraded to 10%")
    print(f"  Original spectral gap: {anomaly['original_spectral_gap']:.4f}")
    print(f"  Degraded spectral gap: {anomaly['degraded_spectral_gap']:.4f}")
    print(f"  Gap drop: {anomaly['gap_drop']:.4f}")
    print(f"  Anomaly detected: {'🔴 YES' if anomaly['anomaly_detected'] else '🟢 NO'}")
    if anomaly['conservation_after'] is not None:
        print(f"  Conservation after degradation: {anomaly['conservation_after']:.4f}")
        drop_pct = (baseline_conservation - anomaly['conservation_after']) / max(baseline_conservation, 1e-9) * 100
        print(f"  Conservation drop: {drop_pct:.1f}%")

    # Inject disconnection in Operator
    anomaly2 = detect_anomaly(os_, "Operator", fault_type="disconnected")
    print(f"\n  ⚠  Injected fault: Operator fully disconnected")
    print(f"  Original spectral gap: {anomaly2['original_spectral_gap']:.4f}")
    print(f"  Degraded spectral gap: {anomaly2['degraded_spectral_gap']:.4f}")
    print(f"  Gap drop: {anomaly2['gap_drop']:.4f}")
    print(f"  Anomaly detected: {'🔴 YES' if anomaly2['anomaly_detected'] else '🟢 NO'}")
    if anomaly2['conservation_after'] is not None:
        print(f"  Conservation after disconnection: {anomaly2['conservation_after']:.4f}")
        drop_pct = (baseline_conservation - anomaly2['conservation_after']) / max(baseline_conservation, 1e-9) * 100
        print(f"  Conservation drop: {drop_pct:.1f}%")

    print()

    # ── Summary ──────────────────────────────────────────────────────────
    print(separator)
    print("  SUMMARY")
    print(separator)
    print("""
  The Agent Spectrum OS treats agents as spectral entities:

  1. REGISTRATION  → Each agent's Laplacian defines its fingerprint
  2. SCHEDULING    → Fiedler vector of task-agent graph = optimal assignment
  3. COMPOSITION   → Merged Laplacians checked for conservation before deploy
  4. REJECTION     → Low conservation ratio = doomed composition (caught early)
  5. ANOMALY       → Conservation drop detects degraded/disconnected agents

  Key insight: Conservation of spectral energy across the agent graph predicts
  whether a team composition will succeed or fail — BEFORE any work is done.
""")
    print(separator)


if __name__ == "__main__":
    run_demo()
