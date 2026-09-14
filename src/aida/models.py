"""Data models for Aida climate calculator."""

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
    # Functional requirements derived from usage/users/environment. Drives
    # alternative selection — not a material choice, but constraints that
    # rule out unsuitable materials (e.g. förskole-tambur → halksäker,
    # rengörbar, slittålig mot blöt sand och salt).
    usage_context: str = ""

    def __post_init__(self):
        # Normalize unknown values to "estimated" — keeps UI logic simple
        if self.quantity_source not in ("user_specified", "estimated"):
            self.quantity_source = "estimated"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Component:
        # Tolerant of extra keys from older payloads or future fields.
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            quantity=data.get("quantity", 0),
            unit=data.get("unit", ""),
            category=data.get("category", ""),
            quantity_source=data.get("quantity_source", "estimated"),
            usage_context=data.get("usage_context", ""),
        )


@dataclass
class NeedsAnalysis:
    # Project-level reasoning about user needs. Travels with the Project
    # through the pipeline; consumed by alternatives.py to constrain suggestions
    # to materials suitable for the actual use case (not just CO2-optimal).
    #
    # Split into "what the user said" vs "what the agent inferred" so the user
    # can spot bad inferences before they propagate downstream.
    from_user: str = ""        # Paraphrase of user's input — no agent inference
    inferred: str = ""         # Agent's conclusions about users + environment + functional requirements
    assumptions: list[str] = field(default_factory=list)   # Things the agent assumed without asking — user can correct
    would_clarify: list[str] = field(default_factory=list) # Questions the agent would ask if critical — optional for user

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> NeedsAnalysis:
        if not data:
            return cls()
        return cls(
            from_user=data.get("from_user", ""),
            inferred=data.get("inferred", ""),
            assumptions=list(data.get("assumptions", []) or []),
            would_clarify=list(data.get("would_clarify", []) or []),
        )


@dataclass
class Project:
    building_type: str
    area_bta: float
    components: list[Component] = field(default_factory=list)
    name: str = ""
    description: str = ""
    needs_analysis: NeedsAnalysis = field(default_factory=NeedsAnalysis)

    def to_dict(self) -> dict:
        return {
            "building_type": self.building_type,
            "area_bta": self.area_bta,
            "name": self.name,
            "description": self.description,
            "components": [c.to_dict() for c in self.components],
            "needs_analysis": self.needs_analysis.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> Project:
        components = [Component.from_dict(c) for c in data.get("components", [])]
        return cls(
            building_type=data.get("building_type", ""),
            area_bta=data.get("area_bta", 0),
            name=data.get("name", ""),
            description=data.get("description", ""),
            components=components,
            needs_analysis=NeedsAnalysis.from_dict(data.get("needs_analysis")),
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
    # Exact Boverket product name used as proxy for this component
    # (e.g. "Takduk, PVC" for a vinylgolv). Empty when source is
    # "Uppskattning" (no Boverket proxy available).
    boverket_product: str = ""
    # The conventional standard material the baseline was costed on, named by
    # the baseline agent from building type plus the component's function
    # ("Homogen vinylmatta (PVC)"). This is the answer to "which floor did you
    # cost?" — a golv category aggregate spans a factor of three, so the number
    # on its own does not identify a material.
    assumed_material: str = ""
    # Per-unit figure and its unit. The LLM has always returned both and the
    # dataclass had nowhere to put them, so they were dropped on construction
    # and persisted to Supabase as null. Without them the view can only show a
    # total, and a total cannot be checked against anything.
    co2e_per_unit: float = 0.0
    unit: str = ""
    # Duplicated from the Component on purpose. The baseline view has to show
    # "17,4 kg/m² × 45 m²" for the total to be checkable by hand, and deriving
    # the 45 as co2e_kg / co2e_per_unit reintroduces the rounding that
    # co2e_kg's one decimal already threw away.
    quantity: float = 0.0
    # Where the number came from, for display. Keys when present: kind
    # ("boverket" | "epd_typvärde"), label, level ("subtype" | "category"),
    # subcategory, sample_size, full_median, min, max. A dict rather than seven
    # flat fields because it is display payload that travels together and is
    # shaped differently per kind.
    basis: dict = field(default_factory=dict)

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
                boverket_product=c.get("boverket_product", ""),
                # Defaulted, not required: analyses saved before 2026-09-01 have
                # none of these and must still load.
                assumed_material=c.get("assumed_material", ""),
                co2e_per_unit=c.get("co2e_per_unit") or 0.0,
                unit=c.get("unit", ""),
                basis=c.get("basis") or {},
                quantity=c.get("quantity") or 0.0,
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
    # Units in stock on Palats at analysis time, for reuse listings. cost_sek
    # and co2e_kg are deliberately computed for the FULL component quantity
    # regardless: Aida is an early-planning tool and stock turns over long
    # before anything is procured (Henric, 2026-08-15). This field is what lets
    # the UI show which part of those numbers is an assumption instead of
    # silently implying the stock is there. None for non-reuse alternatives and
    # for analyses saved before 2026-08-15.
    available_quantity: int | None = None
    # How the price was arrived at: "listing" (a real Palats asking price),
    # "market_estimate" (web-searched installed price for the material type),
    # or "" (unknown). Two very different kinds of number that the comparison
    # table used to render identically.
    price_basis: str = ""
    # Which GWP indicator the climate figure rests on: "" (unknown or baseline),
    # "fossil" (the norm, comparable to the Boverket baseline) or "ghg" (total
    # excluding biogenic, used where an EPD's own components did not add up).
    # A GHG figure sits on a different basis than the rest of the table, so it
    # is labelled everywhere it appears.
    gwp_basis: str = ""
    # Clickable deep link to the listing, for reuse rows. The source string
    # ("[Palats] palats.app/listing/<id>") is an identifier the model copies and
    # the id regex reads, not a URL anyone can open. "" for everything else and
    # for analyses saved before 2026-09-14.
    url: str = ""

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
        # Tolerant like the sibling from_dict methods: ignore extra/missing keys
        # from older payloads or future fields instead of raising TypeError.
        return cls(
            components=[
                ComponentSelection(
                    id=c.get("id", ""),
                    name=c.get("name", ""),
                    selected_alternative=c.get("selected_alternative", {}),
                    baseline_co2e_kg=c.get("baseline_co2e_kg", 0),
                    baseline_cost_sek=c.get("baseline_cost_sek", 0),
                )
                for c in data.get("components", [])
            ]
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
    # cost_sek == 0 means "no price found", not "free" — palats_client documents
    # its own price field as "0 if free/unknown" and an EPD-verified alternative
    # that could not be web-priced survives the B1 filter at 0. Summing those as
    # zero kronor turned a partial basket into a fabricated saving, so the totals
    # now carry how much of the basket they actually cover.
    unpriced_components: list[str] = field(default_factory=list)
    # Totals restricted to the components priced on BOTH sides, so a percentage
    # compares like with like instead of a partial sum against a full baseline.
    comparable_cost_sek: float = 0
    comparable_baseline_cost_sek: float = 0

    @property
    def cost_is_partial(self) -> bool:
        return bool(self.unpriced_components)

    def to_dict(self) -> dict:
        return {
            "sammanställning": {
                "total_co2e_kg": self.total_co2e_kg,
                "total_kostnad_sek": self.total_cost_sek,
                "baslinje_total_co2e_kg": self.baseline_total_co2e_kg,
                "baslinje_total_kostnad_sek": self.baseline_total_cost_sek,
                "co2e_besparing_kg": self.co2e_savings_kg,
                "kostnadsskillnad_sek": self.cost_difference_sek,
                "komponenter_utan_pris": self.unpriced_components,
                "jamforbar_kostnad_sek": self.comparable_cost_sek,
                "jamforbar_baslinje_kostnad_sek": self.comparable_baseline_cost_sek,
            },
            "komponenter": self.components,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
