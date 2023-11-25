import argparse
from . import ffmp
from .utils import call_director, process_and_concatenate_clips
from pprint import pprint

def main():
    parser = argparse.ArgumentParser(description="FFMperative CLI tool")

    subparsers_action = parser.add_subparsers(dest="action", help="Top-level actions")

    # Parser for 'do' action
    do_parser = subparsers_action.add_parser("do", help="Run task with ffmp Agent")
    do_parser.add_argument(
        "--url_endpoint",
        default="https://api-inference.huggingface.co/models/bigcode/starcoder",
        help="The url endpoint to use.",
    )
    do_parser.add_argument("--prompt", required=True, help="Prompt to perform a task")

    # Parser for 'compose' action
    compose_parser = subparsers_action.add_parser("compose", help="Compose clips into a video")
    compose_parser.add_argument("--clips", required=True, help="Path to clips directory")
    compose_parser.add_argument("--output", required=False, default="composed_video.mp4", help="Filename for edited video. Default is `composed_video.mp4`")
    compose_parser.add_argument(
        "--url_endpoint",
        default="https://api-inference.huggingface.co/models/bigcode/starcoder",
        help="The url endpoint to use.",
    )

    args = parser.parse_args()

    if args.action == "do":
        results = ffmp(args.prompt, url_endpoint=args.url_endpoint)
        pprint(results)
    elif args.action == "compose":
        compose_plans, join_plan = call_director(args.clips)
        for plan in compose_plans:
            ffmp(plan, url_endpoint=args.url_endpoint)
        results = process_and_concatenate_clips(join_plan, args.output)
        pprint(results)
    else:
        print("Invalid action")

if __name__ == "__main__":
    main()
