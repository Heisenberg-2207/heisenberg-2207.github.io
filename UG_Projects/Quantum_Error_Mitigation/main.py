from scenedetect import SceneManager, open_video, detect, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from utils.segment import segment_video, calculate_ssim, extract_frames, classify_video, classify_all_videos
from models.poseformer.lib.preprocess import h36m_coco_format, revise_kpts
from utils.detection import detect
from models.poseformer.model.poseformer import Model_poseformer
from models.poseformer.common.camera import *
from utils.full_animation import *
from utils.pose_estimationv1 import *
from models.poseformer.vis_poseformer import *
from args.arg import get_args
from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def main():
    args = get_args()

    input_video_path = Path(args.input)
    base_dir = input_video_path.parent

    segments_path = base_dir / "segments"
    bbox_output_dir = base_dir / "bounding_boxes"
    pose_output_dir = base_dir / "pose_output"
    animation_output_dir = base_dir / "animation_output"

    # Ensure all output folders exist
    segments_path.mkdir(exist_ok=True)
    bbox_output_dir.mkdir(exist_ok=True)
    pose_output_dir.mkdir(exist_ok=True)
    animation_output_dir.mkdir(exist_ok=True)

    if args.rally_segment:
        segment_video(str(input_video_path), str(segments_path))
        classify_all_videos(str(segments_path))

    if args.player_det:
        detect(f"{segments_path}/rally_videos", str(bbox_output_dir))

    if args.pose_est:
        # Loop through all .mp4 files in bounding_boxes/
        for video_file in bbox_output_dir.glob("*.mp4"):
            json_file = video_file.with_suffix(video_file.suffix + ".json")
            if json_file.exists():
                print(f"Running pose estimation for {video_file.name}")
                pose_estimation(str(video_file), str(json_file), f"{str(pose_output_dir)}/{video_file.stem}","BO")
                pose_estimation(str(video_file), str(json_file), f"{str(pose_output_dir)}/{video_file.stem}", "BM 1")
                convert_json_to_h36m_npz(str(video_file), f"{str(pose_output_dir)}/{video_file.stem}/pose_estimations_BM 1.json", f"{str(pose_output_dir)}/{video_file.stem}", label="BM 1")
                convert_json_to_h36m_npz(str(video_file), f"{str(pose_output_dir)}/{video_file.stem}/pose_estimations_BO.json", f"{str(pose_output_dir)}/{video_file.stem}", label="BO")
            else:
                print(f"Warning: JSON file missing for {video_file.name}, skipping.")

    if args.animate:
        print("Generating full animation")
        for video_file in bbox_output_dir.glob("*.mp4"):
            print(f"Running full animation for {video_file.name}")
            get_pose3D(str(video_file), f"{str(pose_output_dir)}/{video_file.stem}/keypointsBM 1.npz",f"{str(animation_output_dir)}/{video_file.stem}/batsman")
            get_pose3D(str(video_file), f"{str(pose_output_dir)}/{video_file.stem}/keypointsBO.npz",
                       f"{str(animation_output_dir)}/{video_file.stem}/bowler")

        print('Generating animation successful!')
if __name__ == "__main__":
    main()
