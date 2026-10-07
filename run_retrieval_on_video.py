"""
Script to run LogSTOP for retrieving the top-k videos with segments of length between min_frames and max_frames relevant to a given query.

Usage:
python3.10 run_retrieval_on_video.py --videos_dir path/to/video_database/ \
                                   --query "Always (person)" \
                                   --top_k 5 \
                                   --local_property_predictor yolov8x \
                                   --smoothing_radius 1 \
                                   --batch_size 8 \
                                   --min_frames 10 \
                                   --max_frames 30 \
                                   --device "cuda"
"""

import os
import heapq
from argparse import ArgumentParser
from src.logstop import logstop
from src.preprocess import preprocess_trace
from src.predictors.base import get_local_property_predictor
from src.utils.ltl import parse_formula_from_string, extract_local_properties

if __name__ == "__main__":
    parser = ArgumentParser(description="Run LogSTOP on a video with a local property predictor and an LTL formula.")
    parser.add_argument("--videos_dir", type=str, required=True, help="Path to the directory containing video files.")
    parser.add_argument("--query", type=str, required=True, help="Query (LTL formula) to evaluate.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top videos to retrieve (Default = 5).")
    parser.add_argument("--local_property_predictor", type=str, default="yolov8x", help="Local property predictor model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for local property prediction.")
    parser.add_argument(
        "--smoothing_radius",
        dest="smoothing_radius", type=int, default=0,
        help="Centered smoothing radius for all local properties (default: 0).",
    )
    parser.add_argument("--min_frames", type=int, default=None, help="Minimum number of consecutive frames to consider when computing retrieval score.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of consecutive frames to consider when computing retrieval score.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the local property predictor on (e.g., 'cpu' or 'cuda').")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top_k must be positive")
    if args.min_frames is not None and args.min_frames <= 0:
        parser.error("--min_frames must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max_frames must be positive")
    if (
        args.min_frames is not None
        and args.max_frames is not None
        and args.min_frames > args.max_frames
    ):
        parser.error("--min_frames cannot exceed --max_frames")

    # Step 1: Parse the query
    print(f"Parsing query: {args.query}")
    phi = parse_formula_from_string(args.query)
    local_properties = extract_local_properties(phi)
    print(f"Parsed LTL formula: \"{phi}\" with local properties: {local_properties}")

    # Step 2: Load the local property predictor
    local_predictor = get_local_property_predictor(args.local_property_predictor)

    # Step 3: Collect logstops for all videos
    video_files = [os.path.join(args.videos_dir, f) for f in os.listdir(args.videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    top_k_videos = []

    for video_path in video_files:
        trace = local_predictor.generate_trace(video_path, batch_size=args.batch_size, local_properties=local_properties, device=args.device)
        processed_trace = preprocess_trace(trace, args.smoothing_radius)

        # Step 3.1: Compute LogSTOP score for the video as the maximum over all valid segments of length between min_frames and max_frames
        length_of_trace = len(trace[list(trace.keys())[0]])
        min_frames = args.min_frames if args.min_frames is not None else 1
        max_frames = args.max_frames if args.max_frames is not None else length_of_trace
        
        score = float('-inf')

        for end_idx in range(min_frames - 1, length_of_trace):
            first_start_idx = max(0, end_idx - max_frames + 1)
            last_start_idx = end_idx - min_frames + 1
            memo = {}
            for start_idx in range(first_start_idx, last_start_idx + 1):
                current_score = logstop(
                    processed_trace,
                    phi,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    memo=memo,
                )
                score = max(score, current_score)

        print(f"LogSTOP score for the video at {video_path} with query '{args.query}': {score}")
        item = (score, video_path)
        if len(top_k_videos) < args.top_k:
            heapq.heappush(top_k_videos, item)
        elif item > top_k_videos[0]:
            heapq.heapreplace(top_k_videos, item)

    # Step 4: Retrieve top-k videos
    top_k_videos.sort(reverse=True)
    print(f"Top-{args.top_k} videos for the query '{args.query}':")
    for rank, (score, video_path) in enumerate(top_k_videos, start=1):
        print(f"Rank {rank}: Video: {video_path}, LogSTOP Score: {score}")
