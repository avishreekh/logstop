"""
Script to run LogSTOP for query matching on a video using a local property predictor and an LTL formula.

Usage:
python3.10 run_query_matching_on_video.py --video_path path/to/video.mp4 \
                                   --fps 5 \
                                   --query "Always (person)" \
                                   --local_property_predictor yolov8x \
                                   --downsampling_smoothing_window 5 \
                                   --batch_size 8 
"""

from argparse import ArgumentParser
from src.logstop import logstop, log
from src.predictors.base import get_local_property_predictor
from src.utils.video import extract_frames_from_video
from src.utils.ltl import parse_formula_from_string, extract_local_properties

if __name__ == "__main__":
    parser = ArgumentParser(description="Run LogSTOP on a video with a local property predictor and an LTL formula.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to extract from the video (Default = 5).")
    parser.add_argument("--query", type=str, required=True, help="Query (LTL formula) to evaluate.")
    parser.add_argument("--local_property_predictor", type=str, default="yolov8x", help="Local property predictor model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for local property prediction.")
    parser.add_argument("--downsampling_smoothing_window", "-w", type=int, default=1,  help="Downsampling smoothing window (w) for LogSTOP (Default = 1).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the local property predictor on (e.g., 'cpu' or 'cuda').")
    args = parser.parse_args()

    # Step 1: Extract frames from the video
    print(f"Extracting frames from video: {args.video_path} at {args.fps} FPS")
    video_frames = extract_frames_from_video(args.video_path, fps=args.fps)
    print(f"Successfully extracted {len(video_frames)} frames from the video.")

    # Step 2: Parse the LTL formula
    print(f"Parsing query: {args.query}")
    phi = parse_formula_from_string(args.query)
    local_properties = extract_local_properties(phi)
    print(f"Parsed LTL formula: \"{phi}\" with local properties: {local_properties}")

    # Step 3: Load the local property predictor and generate the trace
    print(f"Generating trace using the local property predictor: {args.local_property_predictor}")
    local_predictor = get_local_property_predictor(args.local_property_predictor)
    trace = local_predictor.generate_trace(video_frames, batch_size=args.batch_size, local_properties=local_properties, device=args.device)
    
    # Step 4: Run LogSTOP on the generated trace and LTL formula
    score = logstop(trace, phi, start_idx=0, end_idx=len(video_frames)-1, w=args.downsampling_smoothing_window, memo={})
    print(f"LogSTOP score for the video with query '{args.query}': {score}")

    # Step 5: Compute the adaptive threshold using a random trace
    random_trace = {
        prop: [0.5 for _ in range(len(trace[prop]))] for prop in trace.keys()
    }
    random_score = logstop(random_trace, phi, start_idx=0, end_idx=len(video_frames)-1, w=args.downsampling_smoothing_window, memo={})
    threshold = min(random_score, log(0.5))
    print(f"Adaptive threshold (min between random trace score and ln(0.5)): {threshold}")

    print(f"Query match: {score > threshold}")