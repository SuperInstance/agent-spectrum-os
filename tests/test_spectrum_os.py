"""Tests for Agent Spectrum Operating System."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from spectrum_os import ConservationAgent, AgentSpectrumOS, detect_anomaly, build_demo_agents


class TestConservationAgent:
    def test_creation(self):
        agent = ConservationAgent(name="TestBot", capabilities={"a": 0.5}, connections={"Other": 0.3})
        assert agent.name == "TestBot"
        assert agent.capabilities["a"] == 0.5

    def test_spectral_fingerprint_keys(self):
        agent = ConservationAgent(name="A", capabilities={"x": 1.0, "y": 0.5}, connections={})
        fp = agent.spectral_fingerprint
        assert "eigenvalues" in fp
        assert "spectral_gap" in fp
        assert "fiedler_value" in fp

    def test_spectral_fingerprint_single_capability(self):
        agent = ConservationAgent(name="A", capabilities={"x": 1.0}, connections={})
        fp = agent.spectral_fingerprint
        assert fp["spectral_gap"] == 0.0  # single node graph

    def test_spectral_fingerprint_no_capabilities(self):
        agent = ConservationAgent(name="A", capabilities={}, connections={})
        fp = agent.spectral_fingerprint
        assert fp["spectral_gap"] == 0.0

    def test_capability_match_exact(self):
        agent = ConservationAgent(name="A", capabilities={"coding": 0.9}, connections={})
        score = agent.capability_match({"requires": {"coding": 0.8}})
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_capability_match_no_requirements(self):
        agent = ConservationAgent(name="A", capabilities={"coding": 0.9}, connections={})
        score = agent.capability_match({})
        assert score == 0.1

    def test_capability_match_missing_cap(self):
        agent = ConservationAgent(name="A", capabilities={"coding": 0.9}, connections={})
        score = agent.capability_match({"requires": {"flying": 0.8}})
        assert score < 0.1


class TestAgentSpectrumOS:
    def test_register_agent(self):
        os_ = AgentSpectrumOS()
        agent = ConservationAgent(name="A", capabilities={"x": 0.5}, connections={})
        os_.register(agent)
        assert "A" in os_.agents

    def test_submit_task(self):
        os_ = AgentSpectrumOS()
        os_.submit_task("test task", [{"name": "sub1", "requires": {"x": 0.5}}])
        assert len(os_.task_queue) == 1

    def test_schedule_empty(self):
        os_ = AgentSpectrumOS()
        assert os_.schedule() == []

    def test_schedule_with_agents(self):
        os_ = AgentSpectrumOS()
        os_.register(ConservationAgent(name="A1", capabilities={"x": 0.9}, connections={}))
        os_.register(ConservationAgent(name="A2", capabilities={"y": 0.8}, connections={}))
        os_.submit_task("test", [{"name": "s1", "requires": {"x": 0.5}}, {"name": "s2", "requires": {"y": 0.5}}])
        results = os_.schedule()
        assert len(results) == 1
        assert "assignment" in results[0]
        assert "conservation" in results[0]

    def test_compose_needs_two_agents(self):
        os_ = AgentSpectrumOS()
        os_.register(ConservationAgent(name="A", capabilities={"x": 0.5}, connections={}))
        result = os_.compose(["A"], {"complexity": 0.5})
        assert result["status"] == "ERROR"

    def test_compose_two_agents(self):
        os_ = AgentSpectrumOS()
        a1 = ConservationAgent(name="A1", capabilities={"x": 0.9}, connections={"A2": 0.8})
        a2 = ConservationAgent(name="A2", capabilities={"y": 0.8}, connections={"A1": 0.8})
        os_.register(a1)
        os_.register(a2)
        result = os_.compose(["A1", "A2"], {"complexity": 0.5})
        assert result["status"] in ("APPROVED", "REJECTED")
        assert "conservation" in result

    def test_conservation_ratio_zero_vector(self):
        L = np.eye(3)
        assert AgentSpectrumOS._conservation_ratio(L, np.zeros(3)) == 0.0

    def test_fiedler_to_assignment(self):
        fiedler = np.array([0.1, -0.1, 0.3, -0.2])
        result = AgentSpectrumOS._fiedler_to_assignment(fiedler, 2, 2)
        assert len(result) == 2


class TestAnomalyDetection:
    def test_detect_degraded(self):
        os_ = AgentSpectrumOS()
        for a in build_demo_agents():
            os_.register(a)
        os_.submit_task("test", [{"name": "s1", "requires": {"coding": 0.5}}])
        result = detect_anomaly(os_, "Builder", "degraded")
        assert result["agent"] == "Builder"
        assert result["fault"] == "degraded"
        assert "original_spectral_gap" in result

    def test_detect_disconnected(self):
        os_ = AgentSpectrumOS()
        for a in build_demo_agents():
            os_.register(a)
        os_.submit_task("test", [{"name": "s1", "requires": {"monitoring": 0.5}}])
        result = detect_anomaly(os_, "Operator", "disconnected")
        assert result["agent"] == "Operator"
        assert result["fault"] == "disconnected"


class TestDemoAgents:
    def test_build_demo_agents(self):
        agents = build_demo_agents()
        assert len(agents) == 5
        names = {a.name for a in agents}
        assert names == {"Analyst", "Builder", "Validator", "Researcher", "Operator"}

    def test_demo_agents_have_capabilities(self):
        agents = build_demo_agents()
        for a in agents:
            assert len(a.capabilities) > 0
            assert len(a.connections) > 0
            for cap, strength in a.capabilities.items():
                assert 0.0 <= strength <= 1.0
