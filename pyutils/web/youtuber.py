'''
Collection of utilities to extract text transcripts from YouTube.
Use these functions in this order:

1. extract_video_id
2. fetch_transcript
3. save_transcript

Happy reading !
'''

import re
from typing import Optional


def extract_video_id(url: str) -> Optional[str]:

    try:

        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})"
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    except Exception as ex:
        raise Exception(f"While getting the 'video_id' this error occured: {ex}")

def fetch_transcript(video_id: str, languages: list = ['en'], return_text_only:bool=True) -> Optional[str]:

    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

    try:

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript      = transcript_list.find_transcript(languages)
        fetched         = transcript.fetch()

        if return_text_only:

            full_transcript = ''
            for snippet in fetched:
                full_transcript += ' ' + snippet.text + ' '

            return full_transcript

        return fetched

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        print(f"Error: {e}")

def save_transcript_to_text(text: str, filename_path: str):

    with open(filename_path, 'w', encoding='utf-8') as f:
        f.write(text)
