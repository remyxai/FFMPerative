from . import ffmp
import argparse
from pprint import pprint

def main():
    parser = argparse.ArgumentParser(description="FFMperative CLI tool")

    # Define top-level actions
    subparsers_action = parser.add_subparsers(dest="action", help="Top-level actions")

    # Define 'chat' action
    classify_parser = subparsers_action.add_parser("chat", help="Chat interface for ffmp")
    classify_parser.add_argument("--url_endpoint", default="https://api-inference.huggingface.co/models/bigcode/starcoder", help="The url endpoint to use.")
    classify_parser.add_argument("--prompt", required=True, help="Prompt to perform a task")


    args = parser.parse_args()

    # TODO: Update argument parsing logic and commands
    if args.action == "chat":
        results = ffmp(args.prompt, url_endpoint=args.url_endpoint)
        pprint(results) 
    else:
        print("Invalid action")

if __name__ == "__main__":
    main()
