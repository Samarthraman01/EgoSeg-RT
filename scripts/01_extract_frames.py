import cv2
import os

def extract_frames(video_path, output_dir, fps_sample=1):
    # create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # open video
    cap = cv2.VideoCapture(video_path)

    # get video properties
    video_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration     = total_frames / video_fps

    print(f"Video FPS:      {video_fps:.1f}")
    print(f"Total frames:   {int(total_frames)}")
    print(f"Duration:       {duration:.1f} seconds")
    print(f"Extracting:     1 frame every {int(video_fps)} frames")

    # how many frames to skip between saves
    interval = int(video_fps / fps_sample)

    frame_number  = 0
    saved_count   = 0

    while True:
        success, frame = cap.read()

        # end of video
        if not success:
            break

        # save every interval-th frame
        if frame_number % interval == 0:
            filename = os.path.join(output_dir,
                f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

            if saved_count % 10 == 0:
                print(f"Saved {saved_count} frames...", flush=True)

        frame_number += 1

    cap.release()
    print(f"\nDone! Saved {saved_count} frames to {output_dir}")

# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extract_frames(
        video_path  = os.path.expanduser("~/egoseg_rt/data/pov_footage/IMG_0551.MOV"),
        output_dir  = os.path.expanduser("~/egoseg_rt/data/frames/"),
        fps_sample  = 1    # 1 frame per second
    )