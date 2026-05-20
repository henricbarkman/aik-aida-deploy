"""Data models for AIda climate calculator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Component:
    id: str
    name: str
    quantity: float
    unit: str
    category: str = ""  # e.g. "golv", "vägg", "installation"
    quantity_source: str = "estimated"  # "user_specified" or "estimated"

    def __post_init__(self):
        # Normalize unknown values to "estimated" — keeps UI logic simple
        if self.quantity_source not in ("user_specified", "estimated"):
            self.quantity_source = "estimated"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Project:
    building_type: str
    area_bta: float
    components: list[Component] = field(default_factory=list)
    name: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "building_type": self.building_type,
            "area_bta": self.area_bta,
            "name": self.name,
            "description": self.description,
            "components": [c.to_dict() for c in self.components],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> Project:
        components = [Component(**c) for c in data.get("components", [])]
        return cls(
            building_type=data.get("building_type", ""),
            area_bta=data.get("area_bta", 0),
            name=data.get("name", ""),
            description=data.get("description", ""),
            components=components,
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> Project:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class BaselineResult:
    component_id: str
    component_name: str
    co2e_kg: float
    cost_sek: float
    method: str = "NollCO2"
    description: str = ""
    source: str = ""
    cost_source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Baseline:
    components: list[BaselineResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"method": "NollCO2", "components": [c.to_dict() for c in self.components]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> Baseline:
        components = []
        for c in data.get("components", []):
            components.append(BaselineResult(
                component_id=c.get("component_id", c.get("id", "")),
                component_name=c.get("component_name", c.get("name", "")),
                co2e_kg=c.get("co2e_kg", c.get("baseline_co2e_kg", 0)),
                cost_sek=c.get("cost_sek", c.get("baseline_cost_sek", 0)),
                method=c.get("method", "NollCO2"),
                description=c.get("description", ""),
                source=c.get("source", ""),
                cost_source=c.get("cost_source", ""),
            ))
        return cls(components=components)

    @classmethod
    def from_json_file(cls, path: str | Path) -> Baseline:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class Alternative:
    name: str
    co2e_kg: float
    cost_sek: float
    source: str
    reasoning: str = ""
    alternative_type: str = ""  # "reuse", "climate_optimized", "baseline", "info"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComponentAlternatives:
    component_id: str
    component_name: str
    baseline_co2e_kg: float
    baseline_cost_sek: float
    alternatives: list[Alternative] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "baseline_co2e_kg": self.baseline_co2e_kg,
            "baseline_cost_sek": self.baseline_cost_sek,
            "alternatives": [a.to_dict() for a in self.alternatives],
        }


@dataclass
class AlternativesResult:
    components: list[ComponentAlternatives] = field(default_factory=list)
    commentary: str = ""

    def to_dict(self) -> dict:
        d = {"components": [c.to_dict() for c in self.components]}
        if self.commentary:
            d["commentary"] = self.commentary
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class ComponentSelection:
    id: str
    name: str
    selected_alternative: dict
    baseline_co2e_kg: float
    baseline_cost_sek: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Selections:
    components: list[ComponentSelection] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> Selections:
        return cls(
            components=[ComponentSelection(**c) for c in data.get("components", [])]
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> Selections:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class AggregateResult:
    total_co2e_kg: float
    total_cost_sek: float
    baseline_total_co2e_kg: float
    baseline_total_cost_sek: float
    co2e_savings_kg: float
    cost_difference_sek: float
    components: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sammanställning": {
                "total_co2e_kg": self.total_co2e_kg,
                "total_kostnad_sek": self.total_cost_sek,
                "baslinje_total_co2e_kg": self.baseline_total_co2e_kg,
                "baslinje_total_kostnad_sek": self.baseline_total_cost_sek,
                "co2e_besparing_kg": self.co2e_savings_kg,
                "kostnadsskillnad_sek": self.cost_difference_sek,
            },
            "komponenter": self.components,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
