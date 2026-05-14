import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse, parse_qs
import re


try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter, JSONFormatter
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
        YouTubeRequestFailed
    )
    _YOUTUBE_API_AVAILABLE = True
except ImportError:
    YouTubeTranscriptApi = None
    TextFormatter = None
    JSONFormatter = None
    TranscriptsDisabled = Exception
    NoTranscriptFound = Exception
    VideoUnavailable = Exception
    YouTubeRequestFailed = Exception
    _YOUTUBE_API_AVAILABLE = False


logger = logging.getLogger(__name__)


class YouTubeTranscriptExtractor:

    EXTENSIONS: Dict[str, str] = {'text': '.txt', 'json': '.json', 'srt': '.srt'}

    def __init__(self):
        pass

    def extract_video_id(self, url: str) -> Optional[str]:
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url

        parsed_url = urlparse(url)

        if parsed_url.netloc in ['www.youtube.com', 'youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query).get('v', [None])[0]
            elif parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/embed/')[1].split('?')[0]
        elif parsed_url.netloc in ['youtu.be', 'www.youtu.be']:
            return parsed_url.path.lstrip('/')
        elif parsed_url.netloc in ['music.youtube.com', 'www.music.youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query).get('v', [None])[0]

        logger.error(f"Could not extract video ID from: {url}")
        return None

    def get_available_transcripts(self, video_id: str) -> Dict:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            available_transcripts = {
                'manual': [],
                'generated': [],
                'translatable': []
            }

            for transcript in transcript_list:
                transcript_info = {
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                }

                if transcript.is_generated:
                    available_transcripts['generated'].append(transcript_info)
                else:
                    available_transcripts['manual'].append(transcript_info)

                if transcript.is_translatable:
                    available_transcripts['translatable'].append(transcript_info)

            return available_transcripts

        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, YouTubeRequestFailed) as e:
            logger.error(f"Error getting transcript list: {e}")
            return {}

    def extract_transcript(
        self,
        videos: Union[str, List[str]],
        language_codes: Optional[List[str]] = None,
        prefer_manual: bool = True
    ) -> Dict[str, Optional[List[Dict]]]:
        if isinstance(videos, str):
            videos = [videos]

        results: Dict[str, Optional[List[Dict]]] = {}
        for video in videos:
            video_id = self.extract_video_id(video)
            if not video_id:
                results[video] = None
                continue
            results[video_id] = self._fetch_transcript(video_id, language_codes)
        return results

    def _fetch_transcript(
        self,
        video_id: str,
        language_codes: Optional[List[str]] = None,
    ) -> Optional[List[Dict]]:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            transcript = None

            if language_codes:
                for lang_code in language_codes:
                    try:
                        transcript = transcript_list.find_transcript([lang_code])
                        logger.info(f"Found transcript in language: {lang_code}")
                        break
                    except NoTranscriptFound:
                        continue

            if not transcript:
                try:
                    transcript = next(iter(transcript_list))
                    logger.info(f"Using available transcript in: {transcript.language}")
                except StopIteration:
                    logger.error(f"No transcripts available for: {video_id}")
                    return None

            transcript_data = transcript.fetch()
            logger.info(f"Successfully extracted transcript with {len(transcript_data)} segments")
            return transcript_data

        except TranscriptsDisabled:
            logger.error(f"Transcripts disabled for video: {video_id}")
        except NoTranscriptFound:
            logger.error(f"No transcript found for video: {video_id}")
        except VideoUnavailable:
            logger.error(f"Video unavailable: {video_id}")
        except YouTubeRequestFailed as e:
            logger.error(f"YouTube request failed for {video_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {video_id}: {e}")

        return None

    def format_transcript(
        self,
        transcript_data: List[Dict],
        format_type: str = 'text'
    ) -> str:
        if format_type == 'text':
            return ' '.join(seg['text'] for seg in transcript_data)
        elif format_type == 'json':
            return json.dumps(transcript_data, indent=2)
        elif format_type == 'srt':
            return self._format_as_srt(transcript_data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _format_as_srt(self, transcript_data: List[Dict]) -> str:
        srt_content = []
        for i, segment in enumerate(transcript_data, 1):
            start_time = self._seconds_to_srt_time(segment['start'])
            end_time = self._seconds_to_srt_time(segment['start'] + segment['duration'])
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(segment['text'])
            srt_content.append("")
        return "\n".join(srt_content)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

    def save_transcript(
        self,
        transcript: str,
        output_path: str,
        format_type: str = 'text'
    ) -> bool:
        try:
            dir_part = os.path.dirname(output_path)
            if dir_part:
                os.makedirs(dir_part, exist_ok=True)

            if not output_path.endswith(self.EXTENSIONS.get(format_type, '.txt')):
                output_path += self.EXTENSIONS.get(format_type, '.txt')

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcript)

            logger.info(f"Transcript saved to: {output_path}")
            return True

        except OSError as e:
            logger.error(f"Error saving transcript: {e}")
            return False


def main():
    if not _YOUTUBE_API_AVAILABLE:
        print("Error: youtube-transcript-api is not installed.")
        print("Install it with: pip install youtube-transcript-api")
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('youtube_transcript.log'),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser(
        description='Extract transcripts from YouTube videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python youtube_transcript.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
            python youtube_transcript.py dQw4w9WgXcQ --language en es --format json
            python youtube_transcript.py "https://youtu.be/dQw4w9WgXcQ" --output transcript.txt
        """
    )

    parser.add_argument('url', nargs='+', help='YouTube URL(s) or video ID(s)')
    parser.add_argument('--language', '-l', nargs='+', default=['en'],
                        help='Preferred language codes (e.g., en es fr)')
    parser.add_argument('--format', '-f', choices=['text', 'json', 'srt'], default='text',
                        help='Output format')
    parser.add_argument('--output', '-o',
                        help='Output file path — only used when a single URL is provided')
    parser.add_argument('--info', '-i', action='store_true',
                        help='Show available transcript information only')
    parser.add_argument('--no-manual-preference', action='store_true',
                        help="Don't prefer manual transcripts over generated ones")

    args = parser.parse_args()

    extractor = YouTubeTranscriptExtractor()

    if args.info:
        for url in args.url:
            video_id = extractor.extract_video_id(url)
            if not video_id:
                logger.error(f"Invalid YouTube URL or video ID: {url}")
                continue
            info = extractor.get_available_transcripts(video_id)
            if info:
                print(f"\n--- {video_id} ---")
                print(json.dumps(info, indent=2))
        sys.exit(0)

    results = extractor.extract_transcript(
        args.url,
        language_codes=args.language,
        prefer_manual=not args.no_manual_preference
    )

    failed = [vid for vid, data in results.items() if data is None]
    if failed:
        for vid in failed:
            logger.error(f"Failed to extract transcript for: {vid}")

    succeeded = {vid: data for vid, data in results.items() if data is not None}
    if not succeeded:
        logger.error("No transcripts extracted")
        sys.exit(1)

    for video_id, transcript_data in succeeded.items():
        formatted_transcript = extractor.format_transcript(transcript_data, args.format)

        if args.output and len(succeeded) == 1:
            output_path = args.output
        else:
            output_path = f"{video_id}_transcript{extractor.EXTENSIONS[args.format]}"

        if extractor.save_transcript(formatted_transcript, output_path, args.format):
            print(f"Transcript saved: {output_path}")
        else:
            logger.error(f"Failed to save transcript for: {video_id}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
