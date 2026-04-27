from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import Design


class BaseParser(ABC):
    def __init__(self, input_path: str) -> None:
        self.input_path = Path(input_path)

    def parse(self) -> Design:
        files = self.collect_files()
        design = Design(name=self.parse_design_name(files))
        self.parse_nodes(files, design)
        self.parse_nets(files, design)
        self.parse_die(files, design)
        return design

    @abstractmethod
    def collect_files(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def parse_design_name(self, files: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse_nodes(self, files: Any, design: Design) -> None:
        raise NotImplementedError

    def parse_nets(self, files: Any, design: Design) -> None:
        return None

    def parse_die(self, files: Any, design: Design) -> None:
        return None
