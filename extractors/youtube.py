"""
YouTube 자막 추출기
- 공식 자막 우선 (수동 > 자동)
- 타임스탬프 매핑
- 품질 검증
"""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re
import json
import subprocess
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class ExtractionResult:
    """추출 결과 데이터 클래스"""
    success: bool
    source_type: str  # 'youtube'
    source_url: str
    video_id: str  # YouTube video ID (임베딩용)
    title: str
    channel: str  # 채널명
    upload_date: str  # 공개일 (YYYYMMDD)
    duration: str
    language: str
    transcript_type: str  # 'manual', 'auto', 'whisper'
    segments: list  # [{start, end, text}]
    full_text: str
    quality_score: int  # 0-100
    warnings: list
    # Vision 분석용 추가 필드
    frames_dir: Optional[str] = None  # 프레임 저장 폴더
    frames: Optional[list] = None     # 프레임 정보 리스트
    downloaded_video: Optional[str] = None  # 다운로드된 비디오 경로


def extract_video_id(url: str) -> Optional[str]:
    """YouTube URL에서 video_id 추출"""
    patterns = [
        r'(?:v=|\/videos\/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v=|&v=)([^#\&\?\n]+)',
        r'^([a-zA-Z0-9_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_video_for_vision(video_id: str, output_dir: str = None) -> tuple[Optional[str], list]:
    """
    프레임 추출을 위해 YouTube 영상 다운로드

    Args:
        video_id: YouTube video ID
        output_dir: 저장 폴더 (기본: temp)

    Returns:
        (video_path, warnings)
    """
    import tempfile
    import os

    warnings = []

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="youtube_vision_")

    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    try:
        print(f"   [Vision] YouTube 영상 다운로드 중...")
        result = subprocess.run(
            [
                'yt-dlp',
                '-f', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
                '-o', output_path,
                '--no-warnings',
                '--quiet',
                f'https://www.youtube.com/watch?v={video_id}'
            ],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=600  # 10분 타임아웃
        )

        if result.returncode == 0 and os.path.exists(output_path):
            print(f"   [Vision] 다운로드 완료: {output_path}")
            return output_path, warnings
        else:
            warnings.append(f"다운로드 실패: {result.stderr}")
            return None, warnings

    except subprocess.TimeoutExpired:
        warnings.append("다운로드 시간 초과 (10분)")
        return None, warnings
    except Exception as e:
        warnings.append(f"다운로드 오류: {str(e)}")
        return None, warnings


def get_video_metadata(video_id: str) -> dict:
    """
    yt-dlp로 YouTube 영상 메타데이터 가져오기

    Returns:
        dict: {'title': str, 'channel': str, 'upload_date': str}
    """
    try:
        # yt-dlp로 메타데이터만 가져오기 (영상 다운로드 안 함)
        result = subprocess.run(
            [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                '--no-warnings',
                f'https://www.youtube.com/watch?v={video_id}'
            ],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )

        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            return {
                'title': data.get('title', f'YouTube Video ({video_id})'),
                'channel': data.get('channel', data.get('uploader', 'Unknown')),
                'upload_date': data.get('upload_date', '')  # YYYYMMDD 형식
            }
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    # 실패 시 기본값 반환
    return {
        'title': f'YouTube Video ({video_id})',
        'channel': 'Unknown',
        'upload_date': ''
    }


def format_timestamp(seconds: float) -> str:
    """초를 HH:MM:SS 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_transcript_with_priority(video_id: str, preferred_lang: str = 'ko') -> tuple:
    """
    우선순위에 따라 자막 가져오기
    1. 수동 자막 (선호 언어)
    2. 수동 자막 (영어)
    3. 자동 생성 자막 (선호 언어)
    4. 자동 생성 자막 (영어)
    """
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        
        # 1. 수동 자막 시도
        try:
            transcript = transcript_list.find_manually_created_transcript([preferred_lang, 'en'])
            return transcript.fetch(), 'manual', transcript.language_code
        except NoTranscriptFound:
            pass
        
        # 2. 자동 생성 자막 시도
        try:
            transcript = transcript_list.find_generated_transcript([preferred_lang, 'en'])
            return transcript.fetch(), 'auto', transcript.language_code
        except NoTranscriptFound:
            pass
        
        # 3. 아무 자막이나 가져오기
        for transcript in transcript_list:
            return transcript.fetch(), transcript.is_generated and 'auto' or 'manual', transcript.language_code
            
    except TranscriptsDisabled:
        return None, 'disabled', None
    except Exception as e:
        return None, 'error', str(e)
    
    return None, 'not_found', None


def calculate_quality_score(segments: list, transcript_type: str) -> tuple[int, list]:
    """
    품질 점수 계산 (0-100)
    """
    warnings = []
    score = 100
    
    # 자막 유형에 따른 기본 점수
    if transcript_type == 'auto':
        score -= 15
        warnings.append("자동 생성 자막 사용 - 오타/오류 가능성 있음")
    
    if not segments:
        return 0, ["자막 없음"]
    
    # 빈 세그먼트 체크
    empty_count = sum(1 for s in segments if not s.get('text', '').strip())
    if empty_count > len(segments) * 0.1:
        score -= 10
        warnings.append(f"빈 세그먼트 {empty_count}개 발견")
    
    # 너무 짧은 세그먼트 체크 (노이즈 가능성)
    noise_patterns = ['[음악]', '[박수]', '[웃음]', '[Music]', '[Applause]']
    noise_count = sum(1 for s in segments if any(p in s.get('text', '') for p in noise_patterns))
    if noise_count > 5:
        warnings.append(f"노이즈 마커 {noise_count}개 (자동 필터링됨)")
    
    # 평균 세그먼트 길이 체크
    avg_length = sum(len(s.get('text', '')) for s in segments) / len(segments)
    if avg_length < 10:
        score -= 10
        warnings.append("평균 세그먼트가 매우 짧음 - 분절 품질 낮음")
    
    return max(0, min(100, score)), warnings


def extract_youtube(
    url: str,
    preferred_lang: str = 'ko',
    with_vision: bool = False,
    vision_method: str = 'scene',
    max_frames: int = 100
) -> ExtractionResult:
    """
    YouTube 영상에서 자막 추출

    Args:
        url: YouTube URL
        preferred_lang: 선호 언어 코드 (기본: ko)
        with_vision: 화면 분석용 프레임 추출 여부
        vision_method: 프레임 추출 방식 ('scene' 또는 'interval')
        max_frames: 최대 프레임 수

    Returns:
        ExtractionResult: 추출 결과
    """
    video_id = extract_video_id(url)

    if not video_id:
        return ExtractionResult(
            success=False,
            source_type='youtube',
            source_url=url,
            video_id='',
            title='',
            channel='',
            upload_date='',
            duration='',
            language='',
            transcript_type='',
            segments=[],
            full_text='',
            quality_score=0,
            warnings=['유효하지 않은 YouTube URL']
        )

    # 영상 메타데이터 가져오기
    metadata = get_video_metadata(video_id)
    
    # 자막 가져오기
    raw_transcript, transcript_type, language = get_transcript_with_priority(video_id, preferred_lang)

    if raw_transcript is None:
        return ExtractionResult(
            success=False,
            source_type='youtube',
            source_url=url,
            video_id=video_id,
            title=metadata['title'],
            channel=metadata['channel'],
            upload_date=metadata['upload_date'],
            duration='',
            language='',
            transcript_type=transcript_type,
            segments=[],
            full_text='',
            quality_score=0,
            warnings=[f'자막을 가져올 수 없음: {transcript_type}']
        )
    
    # 세그먼트 변환 및 노이즈 필터링
    noise_patterns = ['[음악]', '[박수]', '[웃음]', '[Music]', '[Applause]', '[음악재생]']
    segments = []

    for item in raw_transcript:
        # Handle both dict and object formats
        if hasattr(item, 'text'):
            text = item.text.strip()
            start = item.start
            duration = item.duration
        else:
            text = item['text'].strip()
            start = item['start']
            duration = item.get('duration', 0)

        # 노이즈 제거
        if any(p in text for p in noise_patterns):
            continue
        if not text:
            continue

        segments.append({
            'start': format_timestamp(start),
            'start_seconds': start,
            'end': format_timestamp(start + duration),
            'end_seconds': start + duration,
            'text': text
        })
    
    # 전체 텍스트 생성 (타임스탬프 포함)
    full_text_parts = []
    for seg in segments:
        full_text_parts.append(f"[{seg['start']}] {seg['text']}")
    full_text = '\n'.join(full_text_parts)
    
    # 품질 점수 계산
    quality_score, warnings = calculate_quality_score(segments, transcript_type)
    
    # 총 길이 계산
    if segments:
        duration = format_timestamp(segments[-1]['end_seconds'])
    else:
        duration = '00:00'

    # Vision 분석용 프레임 추출
    frames_dir = None
    frames_list = None
    downloaded_video = None

    if with_vision:
        import os
        import tempfile

        try:
            # 영상 다운로드
            temp_dir = tempfile.mkdtemp(prefix=f"youtube_{video_id}_")
            downloaded_video, download_warnings = download_video_for_vision(video_id, temp_dir)
            warnings.extend(download_warnings)

            if downloaded_video:
                # 프레임 추출
                from extractors.frames import extract_frames

                frames_output_dir = os.path.join(temp_dir, "frames")
                frame_result = extract_frames(
                    downloaded_video,
                    method=vision_method,
                    output_dir=frames_output_dir,
                    max_frames=max_frames
                )

                if frame_result.success:
                    frames_dir = frame_result.output_dir
                    frames_list = [
                        {
                            'path': f.frame_path,
                            'timestamp': f.timestamp,
                            'timestamp_str': f.timestamp_str,
                            'scene_score': f.scene_score
                        }
                        for f in frame_result.frames
                    ]
                    print(f"   [Vision] {len(frames_list)}개 프레임 추출 완료")
                else:
                    warnings.append(f"프레임 추출 실패: {frame_result.warnings}")

        except ImportError as e:
            warnings.append(f"프레임 추출 모듈 로드 실패: {str(e)}")
        except Exception as e:
            warnings.append(f"Vision 처리 오류: {str(e)}")

    return ExtractionResult(
        success=True,
        source_type='youtube',
        source_url=url,
        video_id=video_id,
        title=metadata['title'],
        channel=metadata['channel'],
        upload_date=metadata['upload_date'],
        duration=duration,
        language=language or '',
        transcript_type=transcript_type,
        segments=segments,
        full_text=full_text,
        quality_score=quality_score,
        warnings=warnings,
        frames_dir=frames_dir,
        frames=frames_list,
        downloaded_video=downloaded_video
    )


def to_json(result: ExtractionResult) -> str:
    """결과를 JSON 문자열로 변환"""
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='YouTube 자막 추출')
    parser.add_argument('url', nargs='?', help='YouTube URL')
    parser.add_argument('--lang', '-l', default='ko', help='선호 언어 (기본: ko)')
    parser.add_argument('--with-vision', action='store_true',
                       help='화면 분석용 프레임 추출')
    parser.add_argument('--vision-method', choices=['scene', 'interval'],
                       default='scene', help='프레임 추출 방식 (기본: scene)')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='최대 프레임 수 (기본: 100)')

    args = parser.parse_args()

    if not args.url:
        parser.print_help()
        sys.exit(1)

    result = extract_youtube(
        args.url,
        preferred_lang=args.lang,
        with_vision=args.with_vision,
        vision_method=args.vision_method,
        max_frames=args.max_frames
    )
    print(to_json(result))
