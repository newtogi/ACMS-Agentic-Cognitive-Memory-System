"""
Agent Brain - L1 Perception Gate Module

A modular perception layer that classifies incoming data and determines
salience for downstream processing by the agent brain.
"""

from .perception_gate import perceive, PerceptionResult

__all__ = ["perceive", "PerceptionResult"]
