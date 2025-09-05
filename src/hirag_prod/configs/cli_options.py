import argparse
from argparse import ArgumentParser, Namespace
from typing import Any, Dict


class CliOptions:
    def __init__(self) -> None:
        parser: ArgumentParser = argparse.ArgumentParser(description="HiRAG")
        parser.add_argument("--debug", action="store_true")

        args: Namespace = parser.parse_args()

        self.debug: bool = args.debug

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
