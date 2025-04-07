import base64
import time
from io import BytesIO
from openai import OpenAI
from pathlib import Path
from scenedetect import (
    detect,
    AdaptiveDetector,
    split_video_ffmpeg,
)
import re
import mimetypes
import pprint
import subprocess
from imageio_ffmpeg import read_frames, get_ffmpeg_exe
import json
import pysrt

from google import genai
from google.genai import types
from pydantic import BaseModel
from pymediainfo import MediaInfo

from lib.ytdlp import YouTubeDownloader


def extract_captions_to_srt(response_text, video_path=None):
    """
    Process the JSON caption result and convert it to standard SRT format using pysrt

    Args:
        response_text (str or dict): JSON string or dict containing caption data
        video_path (str): Path to the video file

    Returns:
        str: Formatted SRT content string
    """
    # Parse JSON from string if it's a string
    if isinstance(response_text, str):
        try:
            pprint.pprint(response_text)
            captions = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return ""

    else:
        # If it's already a dict/list, use it directly
        captions = response_text

    # Create a new SubRipFile
    srt_file = pysrt.SubRipFile()

    for caption in captions:
        # Extract caption data
        index = caption.get("index", 0)
        start_str = caption.get("start", "00:00,000")
        end_str = caption.get("end", "00:00,000")
        text = caption.get("text", "")

        try:
            print(
                f"Processing time formats - start: '{start_str}', end: '{end_str}'")

            # Convert timestamp to standard SRT format: HH:MM:SS,mmm
            def normalize_timestamp(timestamp):
                # Replace dots with commas for milliseconds
                if "." in timestamp:
                    timestamp = timestamp.replace(".", ",")

                # Check for two colons (common typo in timestamps)
                if timestamp.count(":") == 2:
                    # Find the position of the second colon
                    first_colon = timestamp.find(":")
                    second_colon = timestamp.find(":", first_colon + 1)

                    # Replace the second colon with a comma
                    timestamp = (
                        timestamp[:second_colon] + "," +
                        timestamp[second_colon + 1:]
                    )

                time_part, ms_part = timestamp.split(",")
                minutes, seconds = time_part.split(":")

                # Ensure proper formatting
                minutes = minutes.zfill(2)  # Pad minutes to 2 digits
                seconds = seconds.zfill(2)  # Pad seconds to 2 digits
                ms_part = ms_part.ljust(3, "0")  # Pad milliseconds to 3 digits

                # Return properly formatted timestamp
                print(
                    f"Corrected timestamp with two colons: 00:{minutes}:{seconds},{ms_part}"
                )
                return f"00:{minutes}:{seconds},{ms_part}"

            # Apply normalization to both timestamps
            start_str = normalize_timestamp(start_str)
            end_str = normalize_timestamp(end_str)

            print(
                f"Normalized timestamps - start: '{start_str}', end: '{end_str}'")

            start = pysrt.SubRipTime.from_string(start_str)
            end = pysrt.SubRipTime.from_string(end_str)

            srt_file.append(
                pysrt.SubRipItem(index=index, start=start, end=end, text=text)
            )
        except Exception as e:
            print(f"Error processing caption: {e}")
            print(
                f"Problematic caption data: index={index}, start={start_str}, end={end_str}"
            )
            # 继续处理下一个字幕，不中断整个过程

    return srt_file


class CaptionAPI:
    """
    Class for handling API operations related to image captioning
    """

    def __init__(
        self, api_endpoint, api_key, model_name, max_retries=10, retry_wait_seconds=1
    ):
        """
        Initialize the CaptionAPI

        Args:
            api_endpoint (str): The API endpoint URL
            api_key (str): API key for authentication
            model_name (str): The model to use for captioning
            max_retries (int): Maximum number of retry attempts
            retry_wait_seconds (int): Base seconds to wait between retries
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds

        # Initialize OpenAI client
        self.client = OpenAI(base_url=api_endpoint, api_key=api_key)

    def _image_to_base64(self, img):
        """
        Convert PIL image to base64 string

        Args:
            img: PIL Image object

        Returns:
            str: Base64 encoded image string
        """
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def call_vision_api(self, img_base64, prompt):
        """
        Make an API call to the vision model

        Args:
            img_base64 (str): Base64 encoded image string
            prompt (str): Text prompt for the image

        Returns:
            str: Text response from the API

        Raises:
            Exception: If the API call fails
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ],
                },
            ],
        )

        # Extract text from response
        return response.choices[0].message.content

    def process_image(self, img, img_id, prompt, extract_captions_func):
        """
        Process a single image with the API

        Args:
            img: PIL Image object
            img_id: Identifier for the image
            prompt: The prompt to use for captioning
            extract_captions_func: Function to extract captions from API response

        Returns:
            dict: Results containing captions, or None if failed
        """
        img_base64 = self._image_to_base64(img)
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                # Call the vision API
                text = self.call_vision_api(img_base64, prompt)

                # Extract captions using the provided function
                long, short = extract_captions_func(text)

                if not long or not short:
                    print(
                        f"⚠️ Warning: Unable to extract captions for image {img_id}")
                    return None

                return {"caption_long": long, "caption": short}

            except Exception as try_e:
                retry_count += 1
                if retry_count <= self.max_retries:
                    # Exponential backoff: wait time increases with each retry
                    current_wait_time = self.retry_wait_seconds * (
                        2 ** (retry_count - 1)
                    )
                    print(
                        f"⚠️ Attempt {retry_count}/{self.max_retries} failed for image {img_id}: {str(try_e)}. "
                        f"Retrying in {current_wait_time} seconds..."
                    )
                    time.sleep(current_wait_time)
                else:
                    print(
                        f"❌ Error processing image {img_id} after {self.max_retries} retries: {str(try_e)}"
                    )
                    return None

        return None

    def batch_process_images(self, imgs, ids, prompt, extract_captions_func):
        """
        Process multiple images with the API

        Args:
            imgs: List of PIL Image objects
            ids: List of image identifiers
            prompt: The prompt to use for captioning
            extract_captions_func: Function to extract captions from API response

        Returns:
            dict: Results containing captions for each image
        """
        result = {}
        start_time = time.time()

        for i, (img_id, img) in enumerate(zip(ids, imgs)):
            try:
                caption_result = self.process_image(
                    img, f"{i+1}/{len(imgs)}", prompt, extract_captions_func
                )
                if caption_result:
                    result[img_id] = caption_result
            except Exception as e:
                print(f"❌ Error processing image {i+1}/{len(imgs)}: {str(e)}")

        end_time = time.time()
        c, t = len(imgs), end_time - start_time
        r = t / c if c != 0 else 0
        print(f"📸 Caption {c} images in {t:.2f} seconds, {r:.2f} sec/img.")

        return result


class CaptionVideoAPI(CaptionAPI):
    """
    Class for handling API operations related to video captioning
    Inherits from CaptionAPI and extends functionality for video processing
    """

    def __init__(
        self,
        api_key,
        model_name,
        api_endpoint="https://api.openai.com/v1",
        max_retries=10,
        retry_wait_seconds=1,
        adaptive_threshold=3,
        min_scene_len=15,
    ):
        """
        Initialize the CaptionVideoAPI

        Args:
            api_endpoint (str): The API endpoint URL
            api_key (str): API key for authentication
            model_name (str): The model to use for captioning
            max_retries (int): Maximum number of retry attempts
            retry_wait_seconds (int): Base seconds to wait between retries
        """
        super().__init__(
            api_endpoint, api_key, model_name, max_retries, retry_wait_seconds
        )
        self.genai_config = None
        if "gemini" in model_name:

            class SubtitleItem(BaseModel):
                index: int
                start: str  # Format: "MM:SS,mmm"
                end: str  # Format: "MM:SS,mmm"
                text: str

            system_prompt = load_all_prompts("prompts")["video_gemini"]

            self.client = genai.Client(api_key=api_key)
            self.genai_config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_output_tokens=65536,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
                response_mime_type="application/json",
                response_schema=list[SubtitleItem],
            )
        elif "qwen" in model_name:
            import dashscope

            self.client = dashscope.Client(api_key=api_key)
            # to do
        elif "step" in model_name:
            self.client = OpenAI(base_url=api_endpoint, api_key=api_key)
            # to do

        self.detector = AdaptiveDetector(adaptive_threshold, min_scene_len)
        self.scene_list = []

    def get_video_duration(self, file_path):
        """
        Get the duration of a video file

        Args:
            file_path: Video file path

        Returns:
            float: Video duration in milliseconds
        """
        for track in MediaInfo.parse(file_path).tracks:
            if track.track_type == "Video":
                return track.duration
            elif track.track_type == "Audio":
                return track.duration
        return 0

    def _extract_video_path(self, download_result):
        """
        Extract video path from download result

        Args:
            download_result: Either a download result dict or direct file path

        Returns:
            str: Path to the video file
        """
        if isinstance(download_result, dict):
            return next(
                (
                    f["path"]
                    for f in download_result["files"].values()
                    if f["type"] == "video"
                ),
                None,
            )
        return download_result

    def call_vision_api(self, uploaded_files, prompt, mime_type):
        """
        Make an API call to the vision model

        Args:
            uploaded_files (list): List of uploaded files to process
            prompt (str): Text prompt for the video

        Returns:
            str: Text response from the API

        Raises:
            Exception: If the API call fails
        """
        for attempt in range(self.max_retries):
            try:
                if "gemini" in self.model_name:
                    response = self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=[
                            types.Part.from_uri(
                                file_uri=uploaded_files[0].uri, mime_type=mime_type
                            ),
                            types.Part.from_text(text=prompt),
                        ],
                        config=self.genai_config,
                    )
                    full_response_text = ""
                    for chunk in response:
                        if (
                            not chunk.candidates
                            or not chunk.candidates[0].content
                            or not chunk.candidates[0].content.parts
                        ):
                            continue
                        if chunk.text:
                            print(chunk.text, end="")
                            full_response_text += chunk.text

                    response_json = full_response_text
                    return response_json
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {self.retry_wait_seconds} seconds...")
                    # Exponential backoff: wait time increases with each retry
                    current_wait_time = self.retry_wait_seconds * (2**attempt)
                    time.sleep(current_wait_time)
                    continue
                else:
                    print(
                        f"Failed to generate captions after {self.max_retries} attempts. Skipping."
                    )
                    return ""

    def upload_file(self, path, mime_type=None):
        upload_success = False
        files = []
        file = types.File()

        for upload_attempt in range(self.max_retries):
            try:
                print(f"checking files for: {path}")
                try:
                    file = self.client.files.get(
                        name=sanitize_filename(Path(path).name)
                    )
                    if file.size_bytes == Path(path).stat().st_size:
                        print(f"File {file.name} is already at {file.uri}")
                        files = [file]
                        wait_for_files_active(self.client, files)
                        print(
                            f"File {file.name} is already active at {file.uri}")
                        upload_success = True
                        break
                    else:
                        print(
                            f"File {file.name} is already exist but size not match")
                        client.files.delete(
                            name=sanitize_filename(Path(path).name))
                        raise Exception("Delete same name file and retry")

                except Exception as e:
                    print(f"File {Path(path).name} is not exist")
                    print(f"uploading files for: {path}")
                    try:
                        files = [upload_to_gemini(
                            self.client, path, mime_type)]
                    except Exception as uploade:
                        files = [
                            upload_to_gemini(
                                self.client,
                                path,
                                mime_type,
                                name=f"{Path(path).name}_{int(time.time())}",
                            )
                        ]
                    wait_for_files_active(self.client, files)
                    upload_success = True
                    break

            except Exception as e:
                print(
                    f"Upload attempt {upload_attempt + 1}/{self.max_retries} failed: {e}"
                )
                if upload_attempt < self.max_retries - 1:
                    current_wait_time = self.retry_wait_seconds * (
                        2 ** (upload_attempt - 1)
                    )
                    print(f"Retrying in {current_wait_time} seconds...")
                    time.sleep(current_wait_time)
                else:
                    print("All upload attempts failed")
                    return ""

        if not upload_success:
            print("Failed to upload file")
            return ""
        return files

    def preprocess_video_split(self, uri):
        """
        Preprocess a video by splitting it into scenes

        Args:
            uri (str): URI of the video file

        Returns:
            list: List of paths to the processed video files
        """
        output_dir = Path(uri).parent / f"{Path(uri).stem}_scenes"
        if not output_dir.exists():
            output_dir.mkdir()
            self.scene_list = detect(uri, self.detector, show_progress=True)
            split_video_ffmpeg(
                uri,
                self.scene_list,
                output_dir=output_dir,
                output_file_template="$VIDEO_NAME-$SCENE_NUMBER.mp4",
                show_progress=True,
                show_output=True,
            )
        return [str(p) for p in output_dir.glob("*.mp4")]

    def preprocess_video_segment(self, uri, segment_time=300):
        """
        Preprocess a video by segmenting it into fixed-length chunks

        Args:
            uri (str): URI of the video file
            segment_time (int): Segment length in seconds

        Returns:
            list: List of paths to the processed video files
        """
        gen = read_frames(uri)
        meta = next(gen)
        pprint.pprint(meta)
        if (
            meta.get("duration", 0) > segment_time
        ):  # Check if video duration is greater than 300 seconds (5 minutes)
            # For longer videos, segment into chunks
            print(
                f"Video is longer than {segment_time} seconds, segmenting into chunks"
            )
            output_dir = Path(uri).parent / f"{Path(uri).stem}_segments"

            output_dir.mkdir(exist_ok=True)
            ffmpeg_exe = get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                "-i",
                uri,
                "-f",
                "segment",
                "-c",
                "copy",
                "-segment_time",
                str(segment_time),
                "-reset_timestamps",
                "1",
                "-y",  # 覆盖输出文件
                "-break_non_keyframes",
                "0",
                "-loglevel",
                "verbose",
                f"{output_dir}/{Path(uri).stem}_%03d.mp4",
            ]
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    print(f"Error segmenting video: {stderr}")
                    raise Exception(f"FFmpeg failed: {stderr}")
            except Exception as e:
                print(f"Error segmenting video: {e}")
                raise Exception(f"FFmpeg failed: {stderr}")
            return [str(p) for p in output_dir.glob("*.mp4")]
        else:
            # For shorter videos, just return the original path
            print(
                f"Video is shorter than {segment_time} seconds, donot segment")
            return [uri]

    def process_video(
        self, video_path, prompt, extract_captions_func=extract_captions_to_srt
    ):
        """
        Process a single video with the API

        Args:
            video_path (str): Path to the video file
            prompt (str): The prompt to use for captioning
            extract_captions_func: Function to extract captions from API response

        Returns:
            dict: Results containing captions, or None if failed
        """
        mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
        upload_files = self.upload_file(video_path, mime_type)

        response_text = self.call_vision_api(upload_files, prompt, mime_type)
        if not response_text:
            return ""
        return (
            extract_captions_func(response_text, video_path)
            if extract_captions_func
            else response_text
        )

    def batch_process_videos(
        self,
        uris,
        prompt,
        split=False,
        segment_time=600,
        extract_captions_func=extract_captions_to_srt,
    ):
        """
        Process multiple videos with the API

        Args:
            uris: List of paths to video files
            prompt: The prompt to use for captioning
            split: Whether to split the video into scenes
            segment_time: Segment length in seconds
            extract_captions_func: Function to extract captions from API response

        Returns:
            dict: Results containing captions for each video
        """
        result = {}
        start_time = time.time()
        downloader = YouTubeDownloader()

        for i, uri in enumerate(uris):
            try:
                print(f"🎥 Processing video {i+1}/{len(uris)}: {uri}")
                video_path = None
                if downloader.is_youtube_url(uri):
                    # Check if a directory with video ID already exists
                    # 这是youtube网址，获取最后的视频ID
                    video_id = downloader.extract_video_id(uri)
                    existing_dir = Path(f"downloads/{video_id}")
                    if existing_dir.exists() and existing_dir.is_dir():
                        print(
                            f"📁 Found existing directory for video {video_id}")
                        video_path = str(
                            next(existing_dir.glob("*.mp4"), None))
                        if video_path:
                            print(f"🎬 Using existing video file: {video_path}")
                        else:
                            print(
                                f"❌ No video file found in directory: {existing_dir}"
                            )
                            download_result = downloader.download_video(uri)
                            video_path = self._extract_video_path(
                                download_result)
                            if not video_path:
                                print(f"❌ Failed to download video {uri}")
                                continue
                    else:
                        # Directory doesn't exist, download the video
                        download_result = downloader.download_video(uri)
                        video_path = self._extract_video_path(download_result)
                        if not video_path:
                            print(f"❌ Failed to download video {uri}")
                            continue
                elif Path(uri).is_file():
                    video_path = uri
                else:
                    print(f"❌ Invalid uri path: {uri}")
                    continue

                # Ensure video_path is not None before proceeding
                if not video_path:
                    print(f"❌ No valid video path for {uri}")
                    continue

                processed_videos = (
                    self.preprocess_video_split(video_path)
                    if split
                    else self.preprocess_video_segment(video_path, segment_time)
                )
                duration = 0
                merged_subs = pysrt.SubRipFile()
                for processed_video in processed_videos:
                    caption_result = self.process_video(
                        processed_video, prompt, extract_captions_func
                    )
                    chunk_subs = caption_result

                    for sub in list(chunk_subs):
                        if sub.start.ordinal > segment_time * 1000:
                            chunk_subs.remove(sub)

                    if len(chunk_subs) > 0:
                        chunk_subs.save(
                            Path(processed_video).with_suffix(".srt"), encoding="utf-8"
                        )

                    if duration > 0:
                        last_duration_minutes = int(duration / 60000)
                        last_duration_seconds = int((duration % 60000) / 1000)
                        last_duration_milliseconds = duration % 1000
                        print(
                            f"⏰ Total duration: {last_duration_minutes}m {last_duration_seconds}s {last_duration_milliseconds}ms"
                        )
                        chunk_subs.shift(
                            minutes=last_duration_minutes,
                            seconds=last_duration_seconds,
                            milliseconds=last_duration_milliseconds,
                        )

                    duration += int(float(self.get_video_duration(processed_video)))

                    if len(chunk_subs) > 0:
                        print(f"✅ Captioned video {i+1}/{len(uris)}: {uri}")
                        merged_subs.extend(chunk_subs)

                merged_subs.clean_indexes()
                merged_subs.save(Path(video_path).with_suffix(
                    ".srt"), encoding="utf-8")
                output = ""
                for i, sub in enumerate(merged_subs, start=1):
                    # 格式: 序号 + 时间戳 + 文本
                    output += f"{i}\n"
                    output += f"{sub.start} --> {sub.end}\n"
                    output += f"{sub.text}\n\n"
                result[uri] = output
            except Exception as e:
                print(f"❌ Error processing video {i+1}/{len(uris)}: {str(e)}")

        end_time = time.time()
        c, t = len(uris), end_time - start_time
        r = t / c if c != 0 else 0
        print(f"🎬 Caption {c} videos in {t:.2f} seconds, {r:.2f} sec/video.")

        return result


def load_all_prompts(tmpl_dir="prompts"):
    prompts = {}
    tmpl_path = Path(tmpl_dir)
    for file_path in tmpl_path.glob("*.txt"):
        prompts[file_path.stem] = file_path.read_text()
    return prompts


def sanitize_filename(name: str) -> str:
    """Sanitizes filenames.

    Requirements:
    - Only lowercase alphanumeric characters or dashes (-)
    - Cannot begin or end with a dash
    - Max length is 40 characters
    """
    # Convert to lowercase and replace non-alphanumeric chars with dash
    sanitized = re.sub(r"[^a-z0-9-]", "-", name.lower())
    # Replace multiple dashes with single dash
    sanitized = re.sub(r"-+", "-", sanitized)
    # Remove leading and trailing dashes
    sanitized = sanitized.strip("-")
    # If empty after sanitization, use a default name
    if not sanitized:
        sanitized = "file"
    # Ensure it starts and ends with alphanumeric character
    if sanitized[0] == "-":
        sanitized = "f" + sanitized
    if sanitized[-1] == "-":
        sanitized = sanitized + "f"
    # If length exceeds 40, keep the first 20 and last 19 chars with a dash in between
    if len(sanitized) > 40:
        # Take parts that don't end with dash
        first_part = sanitized[:20].rstrip("-")
        last_part = sanitized[-19:].rstrip("-")
        sanitized = first_part + "-" + last_part
    return sanitized


def upload_to_gemini(client, path, mime_type=None, name=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    original_name = Path(path).name
    safe_name = sanitize_filename(original_name if name is None else name)

    file = client.files.upload(
        file=path,
        config=types.UploadFileConfig(
            name=safe_name,
            mime_type=mime_type,
            display_name=original_name,
        ),
    )
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def wait_for_files_active(client, files):
    """Waits for the given files to be active.

    Some files uploaded to the Gemini API need to be processed before they can be
    used as prompt inputs. The status can be seen by querying the file's "state"
    field.

    Args:
        client: The Gemini client
        files: List of files to wait for
    """

    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = client.files.get(name=name)
        while file.state.name != "ACTIVE":
            if file.state.name == "FAILED":
                raise Exception(f"File {file.name} failed to process")
            print(".", end="")
            time.sleep(10)
            file = client.files.get(name=name)
        print()  # New line after dots
    print("...all files ready")


def save_srt_file(video_path, srt_content):
    """
    Save SRT content to a file with the same base name as the video

    Args:
        video_path (str): Path to the video file
        srt_content (str or pysrt.SubRipFile): SRT content to save

    Returns:
        str: Path to the saved SRT file
    """
    # Create SRT file path with same name as video
    video_file = Path(video_path)
    srt_file = video_file.with_suffix(".srt")

    # Check if srt_content is already a SubRipFile object
    if isinstance(srt_content, pysrt.SubRipFile):
        # Save directly using pysrt
        srt_content.save(str(srt_file), encoding="utf-8")
    else:
        # If it's a string, write it directly
        srt_file.write_text(srt_content, encoding="utf-8")

    print(f"✅ Saved SRT captions to: {srt_file}")

    return str(srt_file)


def extract_and_save_srt(caption_result, video_path):
    """
    Process caption JSON result, convert to SRT, and save to file

    Args:
        caption_result (str): JSON string containing caption data
        video_path (str): Path to the video file

    Returns:
        dict: Dictionary with 'srt_content' and 'srt_file' keys
    """
    # Convert JSON to SRT using pysrt
    srt_content = extract_captions_to_srt(caption_result)

    # Save the SRT file and get the path
    srt_file = save_srt_file(video_path, srt_content)

    # Handle different return types (pysrt object or string)
    if isinstance(srt_content, pysrt.SubRipFile):
        srt_text = str(srt_content)
    else:
        srt_text = srt_content

    return {
        "srt_content": srt_text,
        "srt_file": srt_file,
        "original": caption_result,
    }


def main():
    """
    Test function for batch video processing with the API
    """
    import os

    # Initialize API with your API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return

    caption_api = CaptionVideoAPI(
        api_key,
        model_name="gemini-2.5-pro-exp-03-25",
        max_retries=3,
        retry_wait_seconds=1,
    )

    # Test videos
    uris = ["https://www.youtube.com/watch?v=M85UvH0TRPc"]

    # Example prompt
    prompt = """Please return your response in Pure SRT json format ONLY.
To specify the timestamps minutes:seconds,milliseconds (MM:SS,mmm) format is used."""

    # Process videos
    results = caption_api.batch_process_videos(
        uris, prompt, segment_time=600, extract_captions_func=extract_captions_to_srt
    )

    # Print results
    print("\nFinal Results:")
    for uri, caption in results.items():
        pprint.pprint(f"\nVideo: {uri}")
        pprint.pprint(f"Caption: {caption}")


if __name__ == "__main__":
    main()
