import subprocess
import os

def download_audio_from_channel(channel_url, title_filter):
    # Get the list of video URLs from the channel
    cmd_list_videos = [
        "yt-dlp",
        "-v",
        "--get-id",
        "--match-title", title_filter,
        "--flat-playlist",
        channel_url
    ]
    print("Fetching list of video URLs from the channel...")
    result = subprocess.run(cmd_list_videos, capture_output=True, text=True)
    video_ids = result.stdout.strip().split("\n")

    if not video_ids or not video_ids[0]:
        print("No matching videos found!")
        return

    print(f"Found {len(video_ids)} matching videos.")
    
    # For each video ID, download the audio
    for idx, video_id in enumerate(video_ids, 1):
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        cmd_download_audio = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "wav",
            "--output", "%(title)s.%(ext)s",
            video_url
        ]
        print(f"Downloading audio for video {idx}/{len(video_ids)}...")
        subprocess.run(cmd_download_audio)

    print("Download complete!")


if __name__ == "__main__":
    CHANNEL_URL = "https://www.youtube.com/@ridescapes"
    TITLE_FILTER = r"\(No Talking, No Music\)"
    
    download_audio_from_channel(CHANNEL_URL, TITLE_FILTER)

