import os
import math
import sys
import argparse

# --- Installation Check ---
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    print("Error: The 'moviepy' library is required but not installed.")
    print("Please install it by running: pip install moviepy")
    sys.exit(1)

def split_video_into_clips(video_path, output_dir, clip_duration=60):
    """
    Splits a single video file into clips of a specified duration.

    Args:
        video_path (str): The full path to the video file to be split.
        output_dir (str): The directory where the clips will be saved.
        clip_duration (int): The duration of each clip in seconds.
    """
    # First, check if the video file actually exists.
    if not os.path.exists(video_path):
        print(f"  [ERROR] Video file not found: {video_path}")
        return

    print(f"  Splitting into {clip_duration}-second clips...")

    try:
        # Use a 'with' statement to ensure the video file is properly closed after use.
        with VideoFileClip(video_path) as video:
            duration = video.duration
            # Calculate the total number of clips to be created.
            num_clips = math.ceil(duration / clip_duration)

            for i in range(num_clips):
                start_time = i * clip_duration
                # The end time is the start of the next clip, or the video's total duration if it's the last clip.
                end_time = min((i + 1) * clip_duration, duration)

                # This check prevents creating tiny or zero-length clips from the very end of the video.
                if start_time >= end_time or (end_time - start_time) < 0.1:
                    continue

                # Create a descriptive filename for the clip.
                # e.g., "original_video_output_synced_clip_001.mp4"
                base_filename = os.path.splitext(os.path.basename(video_path))[0]
                output_filename = f"{base_filename}_clip_{i+1:03d}.mp4"
                output_clip_path = os.path.join(output_dir, output_filename)

                print(f"    - Creating clip: {output_clip_path}")

                # Create the subclip from the original video.
                with video.subclip(start_time, end_time) as subclip:
                    # Write the new clip to a file. We specify codecs for broad compatibility.
                    # Set logger to None to prevent moviepy from printing its own progress bars.
                    subclip.write_videofile(output_clip_path, codec="libx264", audio_codec="aac", logger=None)

        print(f"  Successfully finished splitting {os.path.basename(video_path)}")

    except Exception as e:
        print(f"  [CRITICAL ERROR] Failed to process video {video_path}. Reason: {e}")


def process_all_videos(input_scan_dir, output_parent_dir):
    """
    Traverses directories, finds specific videos, splits them into 1-minute clips,
    and saves them in a mirrored directory structure, including the immediate parent
    of the target folders.

    Args:
        input_scan_dir (str): The specific directory to scan for videos (e.g., /path/to/Dataset/02).
        output_parent_dir (str): The root directory where the output will be saved.
    """
    # 1. Define the main output directory and create it if it doesn't exist.
    output_root = os.path.join(output_parent_dir, "RB_ICRA")
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created main output directory: {output_root}")

    # Extract the last part of the input_scan_dir (e.g., "02")
    # This will be used to prepend to the mirrored path in the output
    dataset_level_folder = os.path.basename(input_scan_dir)

    # 2. Define the target folder names.
    target_folders = {"Front", "Helmet", "Rear", "Aria", "Pupil"}

    print(f"\nStarting to scan directory: {input_scan_dir}\n")
    found_videos_count = 0

    # 3. Use os.walk() to traverse the entire directory tree.
    for dirpath, _, filenames in os.walk(input_scan_dir):
        # Check if the current directory is one of our targets.
        parent_folder_name = os.path.basename(dirpath)
        if parent_folder_name in target_folders:
            for filename in filenames:
                # Check if the file is the video we're looking for.
                if filename.endswith("_synced.mp4"): #filename.endswith("output_synced.mp4") or
                    found_videos_count += 1
                    input_video_path = os.path.join(dirpath, filename)
                    print(f"Found video to process: {input_video_path}")

                    # 4. Create the mirrored output directory structure.
                    # Get the path of the video's folder relative to the input_scan_dir.
                    # This will give paths like "session_id_XYZ/Front"
                    relative_path_from_scan_dir = os.path.relpath(dirpath, input_scan_dir)

                    # Now, combine the dataset_level_folder ("02") with this relative path
                    # This creates the desired "02/session_id_XYZ/Front" structure
                    mirrored_path_component = os.path.join(dataset_level_folder, relative_path_from_scan_dir)

                    # Construct the final output directory path
                    mirrored_output_dir = os.path.join(output_root, mirrored_path_component)

                    # The 'exist_ok=True' flag prevents an error if the directory already exists.
                    os.makedirs(mirrored_output_dir, exist_ok=True)

                    # 5. Call the function to split the found video.
                    split_video_into_clips(input_video_path, mirrored_output_dir)

                    print("-" * 30)

    if found_videos_count == 0:
        print("Scan complete. No videos matching the criteria were found.")
    else:
        print(f"Scan complete. Processed {found_videos_count} video(s).")


def main():
    """
    Main function to parse command-line arguments and start the process.
    """
    description = (
        "Scans a specific input directory for videos named '*undistorted_synced.mp4' "
        "inside specific subfolders ('Front', 'Helmet', 'Rear', 'Aria', 'Pupil'), "
        "splits them into 1-minute clips, and saves them to a new location with a "
        "mirrored folder structure that includes the immediate parent directory of the target folders (e.g., '02')."
    )

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("input_dir", help="The Input Directory Path to scan for videos (e.g., /path/to/Dataset/02).")
    parser.add_argument("output_dir", help="The Output Parent Directory Path to save the clips.")

    args = parser.parse_args()

    # Validate that the provided paths are actual directories.
    if not os.path.isdir(args.input_dir):
        print(f"Error: The specified input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if not os.path.isdir(args.output_dir):
        print(f"Error: The specified output directory does not exist: {args.output_dir}")
        sys.exit(1)

    # Start the main processing function with the validated paths.
    process_all_videos(args.input_dir, args.output_dir)
    print("\nScript finished.")


if __name__ == "__main__":
    main()