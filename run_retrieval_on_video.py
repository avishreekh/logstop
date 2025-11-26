"""
Script to run LogSTOP for retrieving the top-k videos with segments of length between min_frames and max_frames relevant to a given query.

Usage:
python3.10 run_retrieval_on_video.py --videos_dir path/to/video_database/ \
                                   --fps 5 \
                                   --query "Always (person)" \
                                   --top_k 5 \
                                   --local_property_predictor yolov8x \
                                   --downsampling_smoothing_window 5 \
                                   --batch_size 8 \
                                   --min_frames 10 \
                                   --max_frames 30 \
                                   --device "cuda"
"""

import os
from argparse import ArgumentParser
from src.logstop import logstop
from src.predictors.base import get_local_property_predictor
from src.utils.video import extract_frames_from_video
from src.utils.ltl import parse_formula_from_string, extract_local_properties

if __name__ == "__main__":
    parser = ArgumentParser(description="Run LogSTOP on a video with a local property predictor and an LTL formula.")
    parser.add_argument("--videos_dir", type=str, required=True, help="Path to the directory containing video files.")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to extract from the video (Default = 5).")
    parser.add_argument("--query", type=str, required=True, help="Query (LTL formula) to evaluate.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top videos to retrieve (Default = 5).")
    parser.add_argument("--local_property_predictor", type=str, default="yolov8x", help="Local property predictor model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for local property prediction.")
    parser.add_argument("--downsampling_smoothing_window", "-w", type=int, default=1,  help="Downsampling smoothing window (w) for LogSTOP (Default = 1).")
    parser.add_argument("--min_frames", type=int, default=None, help="Minimum number of consecutive frames to consider when computing retrieval score.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of consecutive frames to consider when computing retrieval score.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the local property predictor on (e.g., 'cpu' or 'cuda').")
    args = parser.parse_args()

    # Step 1: Parse the query
    print(f"Parsing query: {args.query}")
    phi = parse_formula_from_string(args.query)
    local_properties = extract_local_properties(phi)
    print(f"Parsed LTL formula: \"{phi}\" with local properties: {local_properties}")

    # Step 2: Load the local property predictor
    local_predictor = get_local_property_predictor(args.local_property_predictor)

    # Step 3: Collect logstops for all videos
    video_files = [os.path.join(args.videos_dir, f) for f in os.listdir(args.videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    video_scores = []

    for video_path in video_files:
        video_frames = extract_frames_from_video(video_path, fps=args.fps)[:50]  # Limit to first 50 frames for efficiency
        trace = local_predictor.generate_trace(video_frames, batch_size=args.batch_size, local_properties=local_properties, device=args.device)

        # Step 4: Compute LogSTOP score for the video as the maximum over all valid segments of length between min_frames and max_frames
        min_frames = args.min_frames if args.min_frames is not None else 1
        max_frames = args.max_frames if args.max_frames is not None else len(video_frames)
        
        score = float('-inf')

        for start_idx in range(0, len(video_frames) - min_frames + 1):  # Ensure at least min_frames
            end_idx = min(len(video_frames), start_idx + max_frames)  # Ensure at most max_frames
            memo = {}
            # LogSTOP for the segment [start_idx, end_idx-1] would compute LogSTOPs for all suffix subsegments within this range
            current_score = logstop(trace, phi, start_idx=start_idx, end_idx=end_idx-1, w=args.downsampling_smoothing_window, memo=memo)
            # Get the maximum score over all valid segments (of length between min_frames and max_frames)
            for s_idx in range(start_idx, end_idx - min_frames + 1, args.downsampling_smoothing_window):
                current_score = max(current_score, memo[(phi, s_idx)])
            score = max(score, current_score)

        print(f"LogSTOP score for the video at {video_path} with query '{args.query}': {score}")
        video_scores.append((video_path, score))

    # Step 5: Retrieve top-k videos
    video_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_videos = video_scores[:args.top_k]
    print(f"Top-{args.top_k} videos for the query '{args.query}':")
    for rank, (video_path, score) in enumerate(top_k_videos, start=1):
        print(f"Rank {rank}: Video: {video_path}, LogSTOP Score: {score}")