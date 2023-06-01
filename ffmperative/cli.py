from . import ffmp
import argparse
from pprint import pprint


def main():
    parser = argparse.ArgumentParser(description="FFMperative CLI tool")

    subparsers_action = parser.add_subparsers(dest="action", help="Top-level actions")

    do_parser = subparsers_action.add_parser("do", help="Run task with ffmp Agent")
    do_parser.add_argument(
        "--url_endpoint",
        default="https://api-inference.huggingface.co/models/bigcode/starcoder",
        help="The url endpoint to use.",
    )
    do_parser.add_argument("--prompt", required=True, help="Prompt to perform a task")

    args = parser.parse_args()

    if args.action == "do":
        results = ffmp(args.prompt, url_endpoint=args.url_endpoint)
        pprint(results)
    else:
        print("Invalid action")

if __name__ == "__main__":
    main()
