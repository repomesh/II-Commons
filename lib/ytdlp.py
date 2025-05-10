import re
import json
import yt_dlp
from datetime import datetime
from pathlib import Path


class YouTubeDownloader:
    def __init__(self, output_dir="downloads"):
        """
        Initialize the downloader

        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # YouTube URL matching pattern
        self.youtube_regex = re.compile(
            r"(?:https?://)?(?:www\.)?"
            r"(?:youtube\.com/(?:watch\?v=|embed/|v/|shorts/)|youtu\.be/)"
            r"([a-zA-Z0-9_-]{11})"
        )

    def is_youtube_url(self, url):
        """
        Check if URL is a YouTube link

        Args:
            url: URL to check

        Returns:
            bool: True if YouTube URL, False otherwise
        """
        return bool(self.youtube_regex.match(url))

    def extract_video_id(self, url):
        """
        Extract YouTube video ID from URL

        Args:
            url: YouTube URL

        Returns:
            str: Video ID or None if not found
        """
        match = self.youtube_regex.search(url)
        if match:
            return match.group(1)
        return None

    def get_video_info(self, url):
        """
        Get video information from URL

        Args:
            url: YouTube URL

        Returns:
            dict: Dictionary containing video information
        """
        if not self.is_youtube_url(url):
            return {"status": "error", "message": f"Not a YouTube URL: {url}"}

        video_id = self.extract_video_id(url)
        video_dir = self.output_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {"status": "success", "info": info}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def download_video(self, url):
        """
        Download YouTube video, subtitles, thumbnail and metadata

        Args:
            url: YouTube URL

        Returns:
            dict: Dictionary containing download information
        """
        if not self.is_youtube_url(url):
            return {"status": "error", "message": f"Not a YouTube URL: {url}"}

        video_id = self.extract_video_id(url)
        video_dir = self.output_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        def download_ranges_callback(info_dict, ydl):
            duration = info_dict.get("duration", 0)
            title = info_dict.get("title", "")
            sections = []
            if duration > 600:
                for i in range(0, duration, 600):
                    section = {
                        "start_time": i,
                        "end_time": i + 600,
                        "title": f"{title}_{i//600}",
                        "index": i // 600 + 1,
                    }
                    sections.append(section)
            sections[-1]["end_time"] = duration
            return sections

        # yt-dlp configuration
        ydl_opts = {
            # Prioritize VP9 codec and 480p resolution
            "format": "bestvideo[vcodec^=vp9][height<=480]+bestaudio/best[vcodec^=vp9][height<=480]",
            "merge_output_format": "mp4",  # Merge to mkv format
            "audioquality": 0,  # Best audio quality
            "writesubtitles": True,  # Download subtitles
            "subtitleslangs": [
                "en",
                "zh-Hans",
                "zh-Hant",
                "ja",
                "ko",
            ],  # Download these languages if available
            "subtitlesformat": "ass/best",  # Subtitle format
            "writethumbnail": True,  # Download thumbnail
            "embedthumbnail": False,  # Don't embed thumbnail in video file
            "outtmpl": str(
                video_dir
                / "%(title)s_%(section_number)s_%(section_start)s-%(section_end)s.%(ext)s"
            ),  # Output filename template with segment info
            "quiet": False,
            "progress": True,
            "writedescription": True,  # Download video description
            "writeinfojson": True,  # Write metadata to JSON file
            # Convert webp thumbnails to jpg for better compatibility
            "postprocessors": [
                {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
                {"key": "FFmpegSubtitlesConvertor", "format": "ass"},
            ],
            "download_ranges": download_ranges_callback,
            "verbose": True,
        }

        result = {
            "video_id": video_id,
            "url": url,
            "download_time": datetime.now().isoformat(),
            "download_path": str(video_dir),
            "files": {},
        }

        try:
            # Download video and metadata
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url)

                # Save important metadata
                metadata = {
                    "title": info.get("title"),
                    "uploader": info.get("uploader"),
                    "upload_date": info.get("upload_date"),
                    "duration": info.get("duration"),
                    "view_count": info.get("view_count"),
                    "like_count": info.get("like_count"),
                    "description": info.get("description"),
                    "categories": info.get("categories"),
                    "tags": info.get("tags"),
                    "thumbnail": info.get(
                        "thumbnail"
                    ),  # URL of the best quality thumbnail
                }

                result["metadata"] = metadata

                # Record downloaded files
                for file_path in video_dir.iterdir():
                    if file_path.is_file():
                        file_type = self._get_file_type(file_path.name)
                        result["files"][file_path.name] = {
                            "path": str(file_path),
                            "type": file_type,
                            "size": file_path.stat().st_size,
                        }

                result["status"] = "success"

        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)

        # Save result information
        with open(video_dir / "download_info.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def download_video_with_time(self, url, start_time, end_time):
        """
        Download YouTube video, subtitles, thumbnail and metadata

        Args:
            url: YouTube URL
            start_time: Start time in seconds
            end_time: End time in seconds
        Returns:
            dict: Dictionary containing download information
        """
        if not self.is_youtube_url(url):
            return {"status": "error", "message": f"Not a YouTube URL: {url}"}
        video_id = self.extract_video_id(url)
        video_dir = self.output_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        def download_ranges_callback(info_dict, ydl):
            sections = [
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "title": f"{info_dict.get('title')}",
                }
            ]
            return sections

        # yt-dlp configuration
        ydl_opts = {
            # Prioritize VP9 codec and 480p resolution
            "format": "bestvideo[vcodec^=vp9][height<=480]+bestaudio/best[vcodec^=vp9][height<=480]",
            "merge_output_format": "mp4",  # Merge to mkv format
            "audioquality": 0,  # Best audio quality
            "writesubtitles": True,  # Download subtitles
            "subtitleslangs": [
                "en",
                "zh-Hans",
                "zh-Hant",
                "ja",
                "ko",
            ],  # Download these languages if available
            "subtitlesformat": "ass/best",  # Subtitle format
            "writethumbnail": False,  # Download thumbnail
            "embedthumbnail": False,  # Don't embed thumbnail in video file
            "outtmpl": str(
                video_dir
                / "%(title)s_%(section_start)s-%(section_end)s.%(ext)s"
            ),  # Output filename template with segment info
            "quiet": False,
            "progress": True,
            "writedescription": False,  # Download video description
            "writeinfojson": True,  # Write metadata to JSON file
            # Convert webp thumbnails to jpg for better compatibility
            "postprocessors": [
                {"key": "FFmpegSubtitlesConvertor", "format": "ass"},
            ],
            "download_ranges": download_ranges_callback,
            "verbose": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.params = {
                    "start_time": start_time,
                    "end_time": end_time,
                }
                info = ydl.extract_info(url)
                return {"status": "success", "info": info}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_file_type(self, filename):
        """Determine file type based on extension"""
        ext = Path(filename).suffix.lower()

        if ext in [".mp4", ".mkv", ".webm", ".flv"]:
            return "video"
        elif ext in [".m4a", ".mp3", ".ogg", ".wav"]:
            return "audio"
        elif ext in [".srt", ".vtt", ".ass"]:
            return "subtitle"
        elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
            return "image"
        elif ext == ".json":
            return "metadata"
        elif ext == ".description":
            return "description"
        else:
            return "other"

    def process_url_list(self, url_list):
        """
        Process URL list and download all YouTube videos

        Args:
            url_list: List of URLs

        Returns:
            list: List of download results
        """
        results = []

        for i, url in enumerate(url_list):
            print(f"\n[{i+1}/{len(url_list)}] Processing: {url}")

            if self.is_youtube_url(url):
                print(f"Downloading YouTube video: {url}")
                result = self.download_video(url)
                results.append(result)
            else:
                print(f"Skipping non-YouTube URL: {url}")
                results.append(
                    {"url": url, "status": "skipped",
                        "message": "Not a YouTube URL"}
                )

        return results


def main():
    # Example URL list
    urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/9bZkp7q19f0",
        "https://www.example.com/not-youtube",
        "https://www.youtube.com/watch?v=C0DPdy98e4c",
    ]

    # URLs can be read from command line arguments or a file
    # Examples:
    # - From file: urls = [line.strip() for line in Path('urls.txt').read_text().splitlines() if line.strip()]
    # - From command line arguments

    downloader = YouTubeDownloader(output_dir="youtube_downloads")
    results = downloader.process_url_list(urls)

    # Print download summary
    print("\n===== Download Summary =====")
    for result in results:
        status = result.get("status", "unknown")
        url = result.get("url", "")

        if status == "success":
            metadata = result.get("metadata", {})
            title = metadata.get("title", "Unknown title")
            print(f"✓ {title} - {url}")
        else:
            message = result.get("message", "")
            print(f"✗ {url} - {message}")

    print(f"\nTotal URLs: {len(urls)}")
    print(
        f"Downloaded: {sum(1 for r in results if r.get('status') == 'success')}")
    print(
        f"Failed/Skipped: {sum(1 for r in results if r.get('status') != 'success')}")


if __name__ == "__main__":
    main()
