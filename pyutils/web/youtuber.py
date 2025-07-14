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
except ImportError:
    print("Error: youtube-transcript-api is not installed.")
    print("Install it with: pip install youtube-transcript-api")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_transcript.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class YouTubeTranscriptExtractor:
    """
    A robust YouTube transcript extractor with error handling and multiple output formats.
    """
    
    def __init__(self):
        self.text_formatter = TextFormatter()
        self.json_formatter = JSONFormatter()
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from various YouTube URL formats.
        
        Args:
            url: YouTube URL or video ID
            
        Returns:
            Video ID string or None if invalid
        """
        # If it's already a video ID (11 characters, alphanumeric and hyphens/underscores)
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url
        
        # Parse different YouTube URL formats
        parsed_url = urlparse(url)
        
        # Standard YouTube URLs
        if parsed_url.netloc in ['www.youtube.com', 'youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query).get('v', [None])[0]
            elif parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/embed/')[1].split('?')[0]
        
        # YouTube short URLs
        elif parsed_url.netloc in ['youtu.be', 'www.youtu.be']:
            return parsed_url.path.lstrip('/')
        
        # YouTube Music URLs
        elif parsed_url.netloc in ['music.youtube.com', 'www.music.youtube.com']:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query).get('v', [None])[0]
        
        logger.error(f"Could not extract video ID from: {url}")
        return None
    
    def get_available_transcripts(self, video_id: str) -> Dict:
        """
        Get information about available transcripts for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with transcript information
        """
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
            
        except Exception as e:
            logger.error(f"Error getting transcript list: {e}")
            return {}
    
    def extract_transcript(
        self, 
        video_id: str, 
        language_codes: Optional[List[str]] = None,
        prefer_manual: bool = True
    ) -> Optional[List[Dict]]:
        """
        Extract transcript from YouTube video.
        
        Args:
            video_id: YouTube video ID
            language_codes: List of preferred language codes (e.g., ['en', 'es'])
            prefer_manual: Whether to prefer manual transcripts over generated ones
            
        Returns:
            List of transcript segments or None if failed
        """
        try:
            # Get transcript list
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find transcript in preferred languages
            transcript = None
            
            if language_codes:
                for lang_code in language_codes:
                    try:
                        transcript = transcript_list.find_transcript([lang_code])
                        logger.info(f"Found transcript in language: {lang_code}")
                        break
                    except NoTranscriptFound:
                        continue
            
            # If no specific language found, try to get any available transcript
            if not transcript:
                try:
                    # Get first available transcript
                    transcript = next(iter(transcript_list))
                    logger.info(f"Using available transcript in: {transcript.language}")
                except StopIteration:
                    logger.error("No transcripts available")
                    return None
            
            # Fetch the transcript
            transcript_data = transcript.fetch()
            
            logger.info(f"Successfully extracted transcript with {len(transcript_data)} segments")
            return transcript_data
            
        except TranscriptsDisabled:
            logger.error("Transcripts are disabled for this video")
        except NoTranscriptFound:
            logger.error("No transcript found for this video")
        except VideoUnavailable:
            logger.error("Video is unavailable")
        except YouTubeRequestFailed as e:
            logger.error(f"YouTube request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        
        return None
    
    def format_transcript(
        self, 
        transcript_data: List[Dict], 
        format_type: str = 'text'
    ) -> str:
        """
        Format transcript data.
        
        Args:
            transcript_data: Raw transcript data
            format_type: 'text', 'json', or 'srt'
            
        Returns:
            Formatted transcript string
        """
        if format_type == 'text':
            return self.text_formatter.format_transcript(transcript_data)
        elif format_type == 'json':
            return self.json_formatter.format_transcript(transcript_data)
        elif format_type == 'srt':
            return self._format_as_srt(transcript_data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _format_as_srt(self, transcript_data: List[Dict]) -> str:
        """
        Format transcript as SRT subtitle format.
        
        Args:
            transcript_data: Raw transcript data
            
        Returns:
            SRT formatted string
        """
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
        """Convert seconds to SRT time format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')
    
    def save_transcript(
        self, 
        transcript: str, 
        output_path: str, 
        video_id: str,
        format_type: str = 'text'
    ) -> bool:
        """
        Save transcript to file.
        
        Args:
            transcript: Formatted transcript content
            output_path: Output file path
            video_id: YouTube video ID
            format_type: Format type for extension
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            
            # Add appropriate extension if not present
            extensions = {'text': '.txt', 'json': '.json', 'srt': '.srt'}
            if not output_path.endswith(extensions.get(format_type, '.txt')):
                output_path += extensions.get(format_type, '.txt')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            logger.info(f"Transcript saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            return False


def main():
    """Main function to run the transcript extractor."""
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
    
    parser.add_argument(
        'url',
        help='YouTube URL or video ID'
    )
    
    parser.add_argument(
        '--language', '-l',
        nargs='+',
        default=['en'],
        help='Preferred language codes (e.g., en es fr)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json', 'srt'],
        default='text',
        help='Output format'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: video_id_transcript.{ext})'
    )
    
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Show available transcript information only'
    )
    
    parser.add_argument(
        '--no-manual-preference',
        action='store_true',
        help='Don\'t prefer manual transcripts over generated ones'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = YouTubeTranscriptExtractor()
    
    # Extract video ID
    video_id = extractor.extract_video_id(args.url)
    if not video_id:
        logger.error("Invalid YouTube URL or video ID")
        sys.exit(1)
    
    logger.info(f"Processing video ID: {video_id}")
    
    # Show transcript info if requested
    if args.info:
        info = extractor.get_available_transcripts(video_id)
        if info:
            print(json.dumps(info, indent=2))
        sys.exit(0)
    
    # Extract transcript
    transcript_data = extractor.extract_transcript(
        video_id,
        language_codes=args.language,
        prefer_manual=not args.no_manual_preference
    )
    
    if not transcript_data:
        logger.error("Failed to extract transcript")
        sys.exit(1)
    
    # Format transcript
    formatted_transcript = extractor.format_transcript(transcript_data, args.format)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        extensions = {'text': '.txt', 'json': '.json', 'srt': '.srt'}
        output_path = f"{video_id}_transcript{extensions[args.format]}"
    
    # Save transcript
    if extractor.save_transcript(formatted_transcript, output_path, video_id, args.format):
        print(f"Transcript successfully saved to: {output_path}")
    else:
        logger.error("Failed to save transcript")
        sys.exit(1)


if __name__ == "__main__":
    main()