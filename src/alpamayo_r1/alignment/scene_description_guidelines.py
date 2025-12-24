# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene Description Guidelines for VLM Training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DescriptionRule:
    """A single scene description guideline rule."""

    id: str
    category: str
    description: str
    priority: str = "medium"
    enabled: bool = True
    details: str = ""

    def to_prompt(self) -> str:
        """Convert rule to a prompt-friendly format."""
        prompt = f"{self.description}"
        if self.details:
            prompt += f"\n{self.details.strip()}"
        return prompt


@dataclass
class SceneDescriptionGuidelines:
    """Container for scene description guideline rules."""

    rules: list[DescriptionRule] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SceneDescriptionGuidelines":
        """Load guidelines from a YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        rules = []
        for rule_data in data.get("rules", []):
            rules.append(DescriptionRule(
                id=rule_data["id"],
                category=rule_data.get("category", "general"),
                description=rule_data["description"],
                priority=rule_data.get("priority", "medium"),
                enabled=rule_data.get("enabled", True),
                details=rule_data.get("details", ""),
            ))

        return cls(rules=rules)

    def get_enabled_rules(self) -> list[DescriptionRule]:
        """Get only enabled rules."""
        return [r for r in self.rules if r.enabled]

    def to_prompt(self) -> str:
        """Convert all enabled rules to a single prompt string for scene description."""
        enabled = self.get_enabled_rules()
        if not enabled:
            return ""

        lines = ["# Scene Description Guidelines\n"]
        lines.append("When describing the off-road scene, follow these principles:\n")

        for i, rule in enumerate(enabled, 1):
            lines.append(f"\n## {i}. {rule.description}")
            if rule.details:
                lines.append(rule.details.strip())

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.rules)


def load_scene_description_guidelines(path: str | Path | None = None) -> SceneDescriptionGuidelines:
    """Load scene description guidelines from path or use default location."""
    if path is None:
        # Default location relative to this file
        path = Path(__file__).parent / "scene_description_guidelines.yaml"
    return SceneDescriptionGuidelines.from_yaml(path)
