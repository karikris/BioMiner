from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="flickr-bio")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("flickr-bio-occurrence 0.1.0")
