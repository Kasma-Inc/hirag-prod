import argparse
from argparse import ArgumentParser, Namespace
from typing import Any, Dict


class CliOptions:
    def __init__(self) -> None:
        parser: ArgumentParser = argparse.ArgumentParser(description="HiRAG")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument(
            "--test",
            default="2",
            help="Test to run. Options: 1/wiki_subcorpus, 2/s3: small_pdf (default), 3/oss: U.S.Health, 4/md-itinerary, 5/md-wiki",
        )

        try:
            args: Namespace = parser.parse_known_args()[0]
        except SystemExit:
            # Fallback if argument parsing fails (e.g., when called from other scripts)
            args = argparse.Namespace(debug=False, test="2")

        self.debug: bool = args.debug
        self.test: str = args.test

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
