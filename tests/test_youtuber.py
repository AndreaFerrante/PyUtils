import pytest
import os


def test_web_init_exports_extractor():
    # Bug: web/__init__.py tries to import nonexistent standalone functions
    # extract_video_id, fetch_transcript, save_transcript_to_text don't exist in youtuber.py
    from pyutils.web import YouTubeTranscriptExtractor
    assert YouTubeTranscriptExtractor is not None


def test_extract_video_id_from_full_url():
    from pyutils.web import YouTubeTranscriptExtractor
    extractor = YouTubeTranscriptExtractor()
    video_id = extractor.extract_video_id('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
    assert video_id == 'dQw4w9WgXcQ'


def test_extract_video_id_from_short_url():
    from pyutils.web import YouTubeTranscriptExtractor
    extractor = YouTubeTranscriptExtractor()
    video_id = extractor.extract_video_id('https://youtu.be/dQw4w9WgXcQ')
    assert video_id == 'dQw4w9WgXcQ'


def test_extract_video_id_from_raw_id():
    from pyutils.web import YouTubeTranscriptExtractor
    extractor = YouTubeTranscriptExtractor()
    video_id = extractor.extract_video_id('dQw4w9WgXcQ')
    assert video_id == 'dQw4w9WgXcQ'


def test_save_transcript_creates_file(tmp_path):
    from pyutils.web import YouTubeTranscriptExtractor
    extractor = YouTubeTranscriptExtractor()
    output_path = str(tmp_path / "transcript")
    result = extractor.save_transcript("Hello world transcript", output_path, 'text')
    assert result is True
    assert os.path.exists(output_path + '.txt')


def test_format_transcript_text():
    from pyutils.web import YouTubeTranscriptExtractor
    extractor = YouTubeTranscriptExtractor()
    data = [{'text': 'Hello world', 'start': 0.0, 'duration': 2.0}]
    result = extractor.format_transcript(data, 'text')
    assert isinstance(result, str)
    assert 'Hello world' in result


def test_format_transcript_srt():
    from pyutils.web import YouTubeTranscriptExtractor
    extractor = YouTubeTranscriptExtractor()
    data = [{'text': 'Hello world', 'start': 0.0, 'duration': 2.0}]
    result = extractor.format_transcript(data, 'srt')
    assert '00:00:00' in result
    assert 'Hello world' in result
