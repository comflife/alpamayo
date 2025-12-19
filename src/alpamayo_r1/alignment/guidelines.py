# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guidelines configuration module for loading and formatting driving rules."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Rule:
    """A single guideline rule."""
    
    id: str
    category: str
    description: str
    priority: str = "medium"
    enabled: bool = True
    details: str = ""
    
    def to_prompt(self) -> str:
        """Convert rule to a prompt-friendly format."""
        prompt = f"[{self.priority.upper()}] {self.description}"
        if self.details:
            prompt += f"\nDetails: {self.details.strip()}"
        return prompt


@dataclass
class Guidelines:
    """Container for all guideline rules."""
    
    rules: list[Rule] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Guidelines":
        """Load guidelines from a YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        rules = []
        for rule_data in data.get("rules", []):
            rules.append(Rule(
                id=rule_data["id"],
                category=rule_data.get("category", "general"),
                description=rule_data["description"],
                priority=rule_data.get("priority", "medium"),
                enabled=rule_data.get("enabled", True),
                details=rule_data.get("details", ""),
            ))
        
        return cls(rules=rules)
    
    def get_enabled_rules(self) -> list[Rule]:
        """Get only enabled rules."""
        return [r for r in self.rules if r.enabled]
    
    def to_prompt(self) -> str:
        """Convert all enabled rules to a single prompt string."""
        enabled = self.get_enabled_rules()
        if not enabled:
            return "No specific guidelines. Follow standard safe driving practices."
        
        lines = ["You must follow these driving guidelines:"]
        for i, rule in enumerate(enabled, 1):
            lines.append(f"\n{i}. {rule.to_prompt()}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.rules)


def load_guidelines(path: str | Path | None = None) -> Guidelines:
    """Load guidelines from path or use default location."""
    if path is None:
        # Default location relative to this file
        path = Path(__file__).parent / "guidelines.yaml"
    return Guidelines.from_yaml(path)
