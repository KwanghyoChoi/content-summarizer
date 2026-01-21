"""
비디오 프레임 추출기
- 장면 변화 감지로 주요 프레임만 추출
- Claude Code Read 도구와 연동하여 화면 분석
- 로컬 비디오 및 YouTube 영상 지원
"""

import os
import cv2
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import tempfile
import shutil


@dataclass
class FrameInfo:
    """추출된 프레임 정보"""
    frame_path: str       # 이미지 파일 경로
    timestamp: float      # 초 단위 타임스탬프
    timestamp_str: str    # HH:MM:SS 형식
    frame_number: int     # 프레임 번호
    scene_score: float    # 장면 변화 점수 (0-1)


@dataclass
class FrameExtractionResult:
    """프레임 추출 결과"""
    success: bool
    frames: list          # FrameInfo 리스트
    output_dir: str       # 프레임 저장 폴더
    total_frames: int     # 추출된 프레임 수
    video_duration: float # 영상 길이 (초)
    warnings: list


def format_timestamp(seconds: float) -> str:
    """초를 HH:MM:SS 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    두 프레임 간의 차이 계산 (0-1 범위)
    높을수록 장면 변화가 큼
    """
    # 그레이스케일 변환
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 리사이즈 (빠른 비교를 위해)
    gray1 = cv2.resize(gray1, (320, 180))
    gray2 = cv2.resize(gray2, (320, 180))

    # 절대 차이 계산
    diff = cv2.absdiff(gray1, gray2)

    # 정규화된 차이 점수 (0-1)
    score = np.mean(diff) / 255.0

    return score


def extract_frames_by_interval(
    video_path: str,
    interval_seconds: float = 30.0,
    output_dir: Optional[str] = None,
    max_frames: int = 100
) -> FrameExtractionResult:
    """
    일정 간격으로 프레임 추출

    Args:
        video_path: 비디오 파일 경로
        interval_seconds: 추출 간격 (초)
        output_dir: 출력 폴더 (None이면 임시 폴더)
        max_frames: 최대 프레임 수

    Returns:
        FrameExtractionResult
    """
    warnings = []

    if not os.path.exists(video_path):
        return FrameExtractionResult(
            success=False,
            frames=[],
            output_dir="",
            total_frames=0,
            video_duration=0,
            warnings=["파일을 찾을 수 없음"]
        )

    # 출력 폴더 설정
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return FrameExtractionResult(
            success=False,
            frames=[],
            output_dir=output_dir,
            total_frames=0,
            video_duration=0,
            warnings=["비디오를 열 수 없음"]
        )

    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frame_count / fps if fps > 0 else 0

    print(f"   비디오 정보: {format_timestamp(duration)}, {fps:.1f}fps")

    frames = []
    frame_interval = int(fps * interval_seconds)
    current_frame = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        if len(frames) >= max_frames:
            warnings.append(f"최대 프레임 수 도달 ({max_frames})")
            break

        timestamp = current_frame / fps
        timestamp_str = format_timestamp(timestamp)

        # 프레임 저장
        frame_filename = f"frame_{len(frames):04d}_{timestamp_str.replace(':', '-')}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        frames.append(FrameInfo(
            frame_path=frame_path,
            timestamp=timestamp,
            timestamp_str=timestamp_str,
            frame_number=current_frame,
            scene_score=0.0
        ))

        current_frame += frame_interval

        if current_frame >= total_frame_count:
            break

    cap.release()

    print(f"   추출 완료: {len(frames)}개 프레임")

    return FrameExtractionResult(
        success=True,
        frames=frames,
        output_dir=output_dir,
        total_frames=len(frames),
        video_duration=duration,
        warnings=warnings
    )


def extract_frames_by_scene_change(
    video_path: str,
    threshold: float = 0.15,
    min_interval: float = 5.0,
    max_interval: float = 60.0,
    output_dir: Optional[str] = None,
    max_frames: int = 100
) -> FrameExtractionResult:
    """
    장면 변화 감지로 프레임 추출

    Args:
        video_path: 비디오 파일 경로
        threshold: 장면 변화 임계값 (0-1, 높을수록 변화가 커야 캡처)
        min_interval: 최소 간격 (초)
        max_interval: 최대 간격 (초) - 변화 없어도 이 간격마다 캡처
        output_dir: 출력 폴더
        max_frames: 최대 프레임 수

    Returns:
        FrameExtractionResult
    """
    warnings = []

    if not os.path.exists(video_path):
        return FrameExtractionResult(
            success=False,
            frames=[],
            output_dir="",
            total_frames=0,
            video_duration=0,
            warnings=["파일을 찾을 수 없음"]
        )

    # 출력 폴더 설정
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return FrameExtractionResult(
            success=False,
            frames=[],
            output_dir=output_dir,
            total_frames=0,
            video_duration=0,
            warnings=["비디오를 열 수 없음"]
        )

    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frame_count / fps if fps > 0 else 0

    print(f"   비디오 정보: {format_timestamp(duration)}, {fps:.1f}fps")
    print(f"   장면 변화 감지 중... (threshold={threshold})")

    frames = []
    min_frame_interval = int(fps * min_interval)
    max_frame_interval = int(fps * max_interval)

    # 첫 프레임 읽기
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return FrameExtractionResult(
            success=False,
            frames=[],
            output_dir=output_dir,
            total_frames=0,
            video_duration=duration,
            warnings=["첫 프레임을 읽을 수 없음"]
        )

    # 첫 프레임 저장
    frame_path = os.path.join(output_dir, "frame_0000_00-00.jpg")
    cv2.imwrite(frame_path, prev_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frames.append(FrameInfo(
        frame_path=frame_path,
        timestamp=0,
        timestamp_str="00:00",
        frame_number=0,
        scene_score=1.0
    ))

    last_capture_frame = 0
    current_frame = min_frame_interval  # 최소 간격부터 시작

    while current_frame < total_frame_count:
        if len(frames) >= max_frames:
            warnings.append(f"최대 프레임 수 도달 ({max_frames})")
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        # 장면 변화 점수 계산
        scene_score = calculate_frame_difference(prev_frame, frame)

        # 캡처 조건 확인
        frames_since_last = current_frame - last_capture_frame
        should_capture = False

        if scene_score >= threshold:
            # 장면 변화 감지
            should_capture = True
        elif frames_since_last >= max_frame_interval:
            # 최대 간격 도달
            should_capture = True
            scene_score = 0.0  # 강제 캡처 표시

        if should_capture:
            timestamp = current_frame / fps
            timestamp_str = format_timestamp(timestamp)

            frame_filename = f"frame_{len(frames):04d}_{timestamp_str.replace(':', '-')}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            frames.append(FrameInfo(
                frame_path=frame_path,
                timestamp=timestamp,
                timestamp_str=timestamp_str,
                frame_number=current_frame,
                scene_score=scene_score
            ))

            last_capture_frame = current_frame
            prev_frame = frame.copy()

            # 진행률 표시
            progress = (current_frame / total_frame_count) * 100
            print(f"\r   진행: {progress:.1f}% ({len(frames)}개 프레임)", end="")

        # 다음 프레임으로 (최소 간격 단위로 이동)
        current_frame += min_frame_interval

    cap.release()
    print(f"\n   추출 완료: {len(frames)}개 프레임 (장면 변화 감지)")

    return FrameExtractionResult(
        success=True,
        frames=frames,
        output_dir=output_dir,
        total_frames=len(frames),
        video_duration=duration,
        warnings=warnings
    )


def extract_frames(
    video_path: str,
    method: str = "scene",
    output_dir: Optional[str] = None,
    **kwargs
) -> FrameExtractionResult:
    """
    프레임 추출 (메인 함수)

    Args:
        video_path: 비디오 파일 경로
        method: 추출 방식 ("scene" 또는 "interval")
        output_dir: 출력 폴더
        **kwargs: 추가 옵션

    Returns:
        FrameExtractionResult
    """
    print(f"   프레임 추출 시작: {os.path.basename(video_path)}")

    if method == "scene":
        return extract_frames_by_scene_change(
            video_path,
            threshold=kwargs.get("threshold", 0.15),
            min_interval=kwargs.get("min_interval", 5.0),
            max_interval=kwargs.get("max_interval", 60.0),
            output_dir=output_dir,
            max_frames=kwargs.get("max_frames", 100)
        )
    else:
        return extract_frames_by_interval(
            video_path,
            interval_seconds=kwargs.get("interval", 30.0),
            output_dir=output_dir,
            max_frames=kwargs.get("max_frames", 100)
        )


def cleanup_frames(output_dir: str) -> None:
    """추출된 프레임 폴더 정리"""
    if os.path.exists(output_dir) and output_dir.startswith(tempfile.gettempdir()):
        shutil.rmtree(output_dir)
        print(f"   임시 폴더 정리 완료: {output_dir}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="비디오 프레임 추출")
    parser.add_argument("video_path", help="비디오 파일 경로")
    parser.add_argument("--method", "-m", choices=["scene", "interval"],
                       default="scene", help="추출 방식")
    parser.add_argument("--threshold", "-t", type=float, default=0.15,
                       help="장면 변화 임계값 (scene 방식)")
    parser.add_argument("--interval", "-i", type=float, default=30.0,
                       help="추출 간격 초 (interval 방식)")
    parser.add_argument("--max-frames", type=int, default=100,
                       help="최대 프레임 수")
    parser.add_argument("--output", "-o", help="출력 폴더")

    args = parser.parse_args()

    result = extract_frames(
        args.video_path,
        method=args.method,
        output_dir=args.output,
        threshold=args.threshold,
        interval=args.interval,
        max_frames=args.max_frames
    )

    if result.success:
        print(f"\n결과:")
        print(f"  - 추출 프레임: {result.total_frames}개")
        print(f"  - 저장 위치: {result.output_dir}")
        print(f"  - 영상 길이: {format_timestamp(result.video_duration)}")

        if result.warnings:
            print(f"  - 경고: {', '.join(result.warnings)}")

        print(f"\n프레임 목록:")
        for f in result.frames[:10]:
            print(f"  [{f.timestamp_str}] score={f.scene_score:.3f} -> {os.path.basename(f.frame_path)}")
        if len(result.frames) > 10:
            print(f"  ... 외 {len(result.frames) - 10}개")
    else:
        print(f"실패: {result.warnings}")
