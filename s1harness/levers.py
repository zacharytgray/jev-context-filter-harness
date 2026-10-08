import tomllib
from dataclasses import dataclass, asdict
from pathlib import Path

SWITCHES = ("relevance", "injection")
# the four configurations, by (relevance, injection). the 50,000-char cap is in every one
NAMES = {(False, False): "plain", (True, False): "relevance", (False, True): "injection",
         (True, True): "relevance-injection"}


@dataclass(frozen=True)
class Levers:
    """a configuration: the plain loop plus the filters it turns on"""
    relevance: bool = False
    injection: bool = False

    @property
    def name(self) -> str:
        """from the switches, never the file name, so an override cannot mislabel a run"""
        return NAMES[(self.relevance, self.injection)]

    def to_dict(self) -> dict:
        return {"name": self.name, **asdict(self)}


def _set(values: dict, section: str, key: str, value):
    if section != "filters" or key not in SWITCHES:
        raise ValueError(f"unknown lever {section}.{key}")
    if not isinstance(value, bool):  # a switch is true or false, never a truthy string
        raise ValueError(f"lever {section}.{key} must be true or false, not {value!r}")
    values[key] = value


def load_levers(path: str | Path | None, overrides: list[str] = ()) -> Levers:
    """a lever file holds one [filters] table with relevance and injection. anything else is an error.
    overrides look like filters.relevance=true, parsed as toml values"""
    values = {}
    if path is not None:
        for section, table in tomllib.loads(Path(path).read_text()).items():
            if not isinstance(table, dict):
                raise ValueError(f"unknown lever {section}")
            for key, value in table.items():
                _set(values, section, key, value)
    for o in overrides:
        dotted, raw = o.split("=", 1)
        section, key = dotted.split(".", 1)
        _set(values, section, key, tomllib.loads(f"v = {raw}")["v"])
    return Levers(**values)
