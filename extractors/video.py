"""
로컬 비디오 파일 자막 추출기
- faster-whisper를 사용한 고속 음성 인식 (기존 대비 4배 빠름)
- 타임스탬프 매핑
- 로컬 파일 메타데이터 추출
- 1~2시간 긴 영상 처리 최적화
"""

import os
import re
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import json

# 사용 가능한 Whisper 모델 (정확도/속도 트레이드오프)
AVAILABLE_MODELS = {
    'tiny': '가장 빠름, 낮은 정확도 (~1GB VRAM)',
    'base': '빠름, 기본 정확도 (~1GB VRAM)',
    'small': '균형잡힌 속도/정확도 (~2GB VRAM)',
    'medium': '높은 정확도, 느림 (~5GB VRAM)',
    'large-v3': '최고 정확도, 가장 느림 (~10GB VRAM)',
}

DEFAULT_MODEL = 'medium'  # RTX 2060 SUPER (8GB)에 적합


@dataclass
class ExtractionResult:
    """추출 결과 데이터 클래스"""
    success: bool
    source_type: str  # 'video'
    source_path: str
    title: str
    creation_date: str  # YYYYMMDD
    duration: str
    language: str
    transcript_type: str  # 'faster-whisper'
    model_used: str  # 사용된 모델
    segments: list  # [{start, end, text}]
    full_text: str
    quality_score: int  # 0-100
    warnings: list
    # Vision 분석용 추가 필드
    frames_dir: Optional[str] = None  # 프레임 저장 폴더
    frames: Optional[list] = None     # 프레임 정보 리스트


def format_timestamp(seconds: float) -> str:
    """초를 HH:MM:SS 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def extract_metadata_from_filename(filepath: str) -> dict:
    """
    파일명에서 메타데이터 추출
    예: 20240521_204707.mp4 -> 2024-05-21 20:47:07
    """
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]

    metadata = {
        'title': name_without_ext,
        'creation_date': ''
    }

    # YYYYMMDD_HHMMSS 패턴 감지
    date_match = re.match(r'^(\d{8})_(\d{6})', name_without_ext)
    if date_match:
        date_str = date_match.group(1)
        time_str = date_match.group(2)
        metadata['creation_date'] = date_str
        metadata['title'] = f"Video_{date_str}_{time_str}"

    return metadata


def transcribe_with_faster_whisper(
    video_path: str,
    language: str = 'ko',
    model_size: str = DEFAULT_MODEL,
    device: str = 'auto'
) -> tuple[list, list, str]:
    """
    faster-whisper를 사용하여 비디오 음성을 텍스트로 변환
    기존 Whisper 대비 4배 빠른 속도

    Args:
        video_path: 비디오 파일 경로
        language: 언어 코드 (기본: ko)
        model_size: 모델 크기 (tiny, base, small, medium, large-v3)
        device: 'auto', 'cuda', 'cpu'

    Returns:
        (segments, warnings, model_used)
    """
    try:
        from faster_whisper import WhisperModel
        from tqdm import tqdm

        # 디바이스 자동 선택
        if device == 'auto':
            try:
                import ctranslate2
                # CUDA 실제 사용 가능 여부 테스트
                cuda_types = ctranslate2.get_supported_compute_types('cuda')
                if cuda_types and 'float16' in cuda_types:
                    # CUDA 라이브러리 존재 여부 추가 확인
                    try:
                        import torch
                        if torch.cuda.is_available():
                            device = 'cuda'
                            compute_type = 'float16'
                        else:
                            device = 'cpu'
                            compute_type = 'int8'
                    except:
                        device = 'cpu'
                        compute_type = 'int8'
                else:
                    device = 'cpu'
                    compute_type = 'int8'
            except:
                device = 'cpu'
                compute_type = 'int8'
        else:
            compute_type = 'float16' if device == 'cuda' else 'int8'

        print(f"   faster-whisper 모델 로딩 중... (model={model_size}, device={device})")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # 비디오 길이 확인 (진행률 표시용)
        duration_seconds = get_video_duration(video_path)
        duration_str = format_timestamp(duration_seconds) if duration_seconds > 0 else "알 수 없음"
        print(f"   비디오 길이: {duration_str}")
        print(f"   음성 인식 중... (긴 영상은 시간이 걸립니다)")

        # 음성 인식 수행 (VAD 필터로 빈 구간 스킵)
        segments_iter, info = model.transcribe(
            video_path,
            language=language,
            vad_filter=True,  # 무음 구간 스킵
            vad_parameters=dict(
                min_silence_duration_ms=500,  # 0.5초 이상 무음 스킵
            ),
            word_timestamps=True,
        )

        segments = []
        warnings = []

        # 진행률 표시와 함께 세그먼트 수집
        with tqdm(total=duration_seconds, unit='sec', desc='   처리 중') as pbar:
            last_end = 0
            for seg in segments_iter:
                text = seg.text.strip()
                if not text:
                    continue

                segments.append({
                    'start': format_timestamp(seg.start),
                    'start_seconds': seg.start,
                    'end': format_timestamp(seg.end),
                    'end_seconds': seg.end,
                    'text': text
                })

                # 진행률 업데이트
                pbar.update(seg.end - last_end)
                last_end = seg.end

            # 남은 진행률 채우기
            if last_end < duration_seconds:
                pbar.update(duration_seconds - last_end)

        if not segments:
            warnings.append("음성 인식 결과 없음")

        # 감지된 언어 정보
        if info.language != language:
            warnings.append(f"감지된 언어: {info.language} (요청: {language})")

        return segments, warnings, model_size

    except ImportError:
        # fallback to original whisper
        print("   faster-whisper 미설치, 기존 Whisper로 fallback...")
        return transcribe_with_original_whisper(video_path, language)
    except Exception as e:
        return [], [f"faster-whisper 오류: {str(e)}"], model_size


def transcribe_with_original_whisper(video_path: str, language: str = 'ko') -> tuple[list, list, str]:
    """
    기존 OpenAI Whisper fallback (slower)
    """
    try:
        # imageio-ffmpeg의 FFmpeg 바이너리 사용
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            if ffmpeg_dir not in os.environ.get('PATH', ''):
                os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
        except ImportError:
            pass

        import whisper

        print("   Whisper 모델 로딩 중... (small)")
        model = whisper.load_model("small")

        print("   음성 인식 중... (시간이 걸릴 수 있습니다)")
        result = model.transcribe(
            video_path,
            language=language,
            verbose=False,
            word_timestamps=True
        )

        segments = []
        warnings = ["기존 Whisper 사용 (faster-whisper 권장)"]

        for seg in result['segments']:
            text = seg['text'].strip()
            if not text:
                continue

            segments.append({
                'start': format_timestamp(seg['start']),
                'start_seconds': seg['start'],
                'end': format_timestamp(seg['end']),
                'end_seconds': seg['end'],
                'text': text
            })

        if not segments:
            warnings.append("음성 인식 결과 없음")

        return segments, warnings, 'small'

    except ImportError:
        return [], ["Whisper 라이브러리 미설치"], 'none'
    except Exception as e:
        return [], [f"Whisper 오류: {str(e)}"], 'none'


def get_video_duration(video_path: str) -> float:
    """비디오 재생 시간 가져오기"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
    except:
        return 0


def calculate_quality_score(segments: list, model_used: str = 'small') -> tuple[int, list]:
    """품질 점수 계산"""
    warnings = []
    score = 100

    if not segments:
        return 0, ["자막 추출 실패"]

    # 모델에 따른 기본 점수 조정
    model_scores = {
        'large-v3': 0,   # 최고 품질
        'medium': -5,
        'small': -10,
        'base': -15,
        'tiny': -20,
    }
    score += model_scores.get(model_used, -10)

    if model_used not in ['large-v3', 'medium']:
        warnings.append(f"{model_used} 모델 사용 - 더 큰 모델 권장")

    # 세그먼트 개수 체크
    if len(segments) < 5:
        score -= 10
        warnings.append("세그먼트가 매우 적음")

    # 평균 길이 체크
    avg_length = sum(len(s['text']) for s in segments) / len(segments)
    if avg_length < 10:
        score -= 10
        warnings.append("평균 세그먼트가 매우 짧음")

    return max(0, min(100, score)), warnings


def extract_video(
    video_path: str,
    language: str = 'ko',
    model_size: str = DEFAULT_MODEL,
    with_vision: bool = False,
    vision_method: str = 'scene',
    max_frames: int = 100
) -> ExtractionResult:
    """
    로컬 비디오 파일에서 음성 추출 및 텍스트 변환

    Args:
        video_path: 비디오 파일 경로
        language: 언어 코드 (기본: ko)
        model_size: Whisper 모델 크기 (tiny, base, small, medium, large-v3)
        with_vision: 화면 분석용 프레임 추출 여부
        vision_method: 프레임 추출 방식 ('scene' 또는 'interval')
        max_frames: 최대 프레임 수

    Returns:
        ExtractionResult: 추출 결과
    """
    if not os.path.exists(video_path):
        return ExtractionResult(
            success=False,
            source_type='video',
            source_path=video_path,
            title='',
            creation_date='',
            duration='',
            language='',
            transcript_type='',
            model_used='',
            segments=[],
            full_text='',
            quality_score=0,
            warnings=['파일을 찾을 수 없음']
        )

    # 파일명에서 메타데이터 추출
    metadata = extract_metadata_from_filename(video_path)

    # 비디오 재생 시간
    duration_seconds = get_video_duration(video_path)
    duration_str = format_timestamp(duration_seconds) if duration_seconds > 0 else "00:00"

    # 긴 영상 안내
    if duration_seconds > 3600:  # 1시간 이상
        hours = duration_seconds / 3600
        print(f"   [INFO] 긴 영상 감지 ({hours:.1f}시간)")
        print(f"   [INFO] 예상 처리 시간: {hours * 0.3:.0f}~{hours * 0.5:.0f}분 (faster-whisper {model_size})")

    # faster-whisper로 음성 인식
    segments, warnings, model_used = transcribe_with_faster_whisper(
        video_path, language, model_size
    )

    if not segments:
        return ExtractionResult(
            success=False,
            source_type='video',
            source_path=video_path,
            title=metadata['title'],
            creation_date=metadata['creation_date'],
            duration=duration_str,
            language=language,
            transcript_type='faster-whisper',
            model_used=model_used,
            segments=[],
            full_text='',
            quality_score=0,
            warnings=warnings + ['음성 인식 실패']
        )

    # 전체 텍스트 생성 (타임스탬프 포함)
    full_text_parts = []
    for seg in segments:
        full_text_parts.append(f"[{seg['start']}] {seg['text']}")
    full_text = '\n'.join(full_text_parts)

    # 품질 점수 계산
    quality_score, quality_warnings = calculate_quality_score(segments, model_used)
    warnings.extend(quality_warnings)

    # 실제 재생 시간 (세그먼트 기준)
    if segments:
        actual_duration = format_timestamp(segments[-1]['end_seconds'])
    else:
        actual_duration = duration_str

    # Vision 분석용 프레임 추출
    frames_dir = None
    frames_list = None

    if with_vision:
        try:
            from extractors.frames import extract_frames
            print(f"\n   [Vision] 프레임 추출 시작...")

            # output 폴더에 프레임 저장
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            frames_output_dir = os.path.join(
                os.path.dirname(video_path),
                f"{base_name}_frames"
            )

            frame_result = extract_frames(
                video_path,
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
                print(f"   [Vision] 저장 위치: {frames_dir}")
            else:
                warnings.append(f"프레임 추출 실패: {frame_result.warnings}")

        except ImportError as e:
            warnings.append(f"프레임 추출 모듈 로드 실패: {str(e)}")
        except Exception as e:
            warnings.append(f"프레임 추출 오류: {str(e)}")

    return ExtractionResult(
        success=True,
        source_type='video',
        source_path=video_path,
        title=metadata['title'],
        creation_date=metadata['creation_date'] or datetime.now().strftime('%Y%m%d'),
        duration=actual_duration,
        language=language,
        transcript_type='faster-whisper',
        model_used=model_used,
        segments=segments,
        full_text=full_text,
        quality_score=quality_score,
        warnings=warnings,
        frames_dir=frames_dir,
        frames=frames_list
    )


def to_json(result: ExtractionResult) -> str:
    """결과를 JSON 문자열로 변환"""
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='로컬 비디오 음성 추출 (faster-whisper)')
    parser.add_argument('video_path', nargs='?', help='비디오 파일 경로')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL,
                       choices=list(AVAILABLE_MODELS.keys()),
                       help=f'Whisper 모델 (기본: {DEFAULT_MODEL})')
    parser.add_argument('--language', '-l', default='ko', help='언어 코드 (기본: ko)')
    parser.add_argument('--list-models', action='store_true', help='사용 가능한 모델 목록')
    parser.add_argument('--with-vision', action='store_true',
                       help='화면 분석용 프레임 추출')
    parser.add_argument('--vision-method', choices=['scene', 'interval'],
                       default='scene', help='프레임 추출 방식 (기본: scene)')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='최대 프레임 수 (기본: 100)')

    args = parser.parse_args()

    if args.list_models:
        print("\n사용 가능한 모델:")
        for model, desc in AVAILABLE_MODELS.items():
            marker = " (기본)" if model == DEFAULT_MODEL else ""
            print(f"  {model:12} - {desc}{marker}")
        print()
        sys.exit(0)

    if not args.video_path:
        parser.print_help()
        sys.exit(1)

    result = extract_video(
        args.video_path,
        args.language,
        args.model,
        with_vision=args.with_vision,
        vision_method=args.vision_method,
        max_frames=args.max_frames
    )
    print(to_json(result))
