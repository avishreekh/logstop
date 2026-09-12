"""
Script to run LogSTOP for query matching on a video using a local property predictor and a temporal query.

Usage:
python3.10 run_query_matching_on_video.py --video_path path/to/video.mp4 \
                                   --query "Always (person)" \
                                   --local_property_predictor yolov8x \
                                   --smoothing_radius 1 \
                                   --local_threshold 0.5 \
                                   --batch_size 8 \
                                   --device "cuda"
"""

from argparse import ArgumentParser
from src.logstop import logstop
from src.preprocess import preprocess_trace
from src.predictors.base import get_local_property_predictor
from src.utils.ltl import parse_formula_from_string, extract_local_properties

if __name__ == "__main__":
    parser = ArgumentParser(description="Run LogSTOP on a video with a local property predictor and an LTL formula.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--query", type=str, required=True, help="Query (LTL formula) to evaluate.")
    parser.add_argument("--local_property_predictor", type=str, default="yolov8x", help="Local property predictor model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for local property prediction.")
    parser.add_argument(
        "--smoothing_radius",
        dest="smoothing_radius", type=int, default=0,
        help="Centered smoothing radius for all local properties (default: 0).",
    )
    parser.add_argument(
        "--local_threshold",
        type=float,
        default=0.5,
        help="Local-property value used to construct the threshold trace (default: 0.5).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the local property predictor on (e.g., 'cpu' or 'cuda').")
    args = parser.parse_args()
    if not 0.0 <= args.local_threshold <= 1.0:
        parser.error("--local_threshold must be between 0 and 1")

    # Step 1: Parse the query as an LTL formula
    print(f"Parsing query: {args.query}")
    phi = parse_formula_from_string(args.query)
    local_properties = extract_local_properties(phi)
    print(f"Parsed LTL formula: \"{phi}\" with local properties: {local_properties}")

    # Step 2: Load the local property predictor and generate the trace
    print(f"Generating trace using the local property predictor: {args.local_property_predictor}")
    local_predictor = get_local_property_predictor(args.local_property_predictor)
    trace = local_predictor.generate_trace(args.video_path, batch_size=args.batch_size, local_properties=local_properties, device=args.device)
    length_of_trace = len(trace[list(trace.keys())[0]])

    # Step 3: Run LogSTOP on the generated trace and LTL formula after preprocessing
    processed_trace = preprocess_trace(trace, args.smoothing_radius)
    score = logstop(processed_trace, phi, start_idx=0, end_idx=length_of_trace-1)
    print(f"LogSTOP score for the video with query '{args.query}': {score}")

    # Step 4: Compute the adaptive threshold using a constant threshold trace
    threshold_trace = {
        prop: [args.local_threshold] * len(values) for prop, values in trace.items()
    }
    processed_threshold_trace = preprocess_trace(
        threshold_trace, args.smoothing_radius
    )
    threshold = logstop(
        processed_threshold_trace, phi, start_idx=0, end_idx=length_of_trace - 1
    )
    print(
        "Adaptive threshold "
        f"(LogSTOP over the {args.local_threshold} threshold trace): {threshold}"
    )

    print(f"Query match: {score > threshold}")
