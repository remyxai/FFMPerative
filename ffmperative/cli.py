import argparse
from . import ffmp
from .utils import call_director, process_and_concatenate_clips
from pprint import pprint

def main():
    parser = argparse.ArgumentParser(description="FFMperative CLI tool")

    subparsers_action = parser.add_subparsers(dest="action", help="Top-level actions")

    # Parser for 'do' action
    do_parser = subparsers_action.add_parser("do", help="Run task with ffmp Agent")
    do_parser.add_argument("--prompt", required=True, help="Prompt to perform a task")
    do_parser.add_argument("--remote", action='store_true', default=False, required=False, help="Run remotely")

    # Parser for 'compose' action
    compose_parser = subparsers_action.add_parser("compose", help="Compose clips into a video")
    compose_parser.add_argument("--clips", required=True, help="Path to clips directory")
    compose_parser.add_argument("--prompt", required=False, default=None, help="Guide the composition by text prompt e.g. 'Edit the video for social media'")
    compose_parser.add_argument("--output", required=False, default="composed_video.mp4", help="Filename for edited video. Default is 'composed_video.mp4'")
    compose_parser.add_argument("--remote", action='store_true', default=False, required=False, help="Run remotely")

    args = parser.parse_args()

    if args.action == "do":
        results = ffmp(args.prompt, args.remote)
        pprint(results)
    elif args.action == "compose":
        compose_plans, join_plan = call_director(args.clips, args.prompt)
        for plan in compose_plans:
            try:
                ffmp(plan, args.remote)                
            except:
                print("plan: ", plan)        
        results = process_and_concatenate_clips(join_plan, args.output)
        pprint(results)
    else:
        print("Invalid action")

if __name__ == "__main__":
    main()
