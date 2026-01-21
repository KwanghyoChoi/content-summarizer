#!/usr/bin/env python3
"""
Content Summarizer - 메인 실행 파일
YouTube, PDF, 웹페이지에서 정확한 요약 노트 생성
"""

import argparse
import os
import sys
import re
from datetime import datetime


def select_formats_interactive() -> list:
    """
    대화형으로 노트 형식 선택
    """
    available_formats = {
        '1': ('detailed', '상세 노트 - 계층적 구조의 포괄적인 노트'),
        '2': ('essence', '핵심 노트 - 5~10개 핵심 포인트'),
        '3': ('easy', '쉬운 노트 - 초보자용 3~5개 핵심'),
        '4': ('mindmap', '마인드맵 - Mermaid 다이어그램 + 트리 구조'),
    }

    print("\n" + "="*50)
    print("노트 형식 선택")
    print("="*50)
    print("\n생성할 노트 형식을 선택하세요:\n")

    for key, (name, desc) in available_formats.items():
        print(f"  [{key}] {name:10} - {desc}")

    print(f"\n  [A] 전체 선택 (All)")
    print(f"  [Q] 취소 (Quit)")

    print("\n" + "-"*50)
    print("입력 예시: 1,2 또는 1 2 또는 134 또는 A")
    print("-"*50)

    while True:
        user_input = input("\n선택: ").strip().upper()

        if user_input == 'Q':
            print("취소되었습니다.")
            return []

        if user_input == 'A':
            selected = [fmt for fmt, _ in available_formats.values()]
            print(f"\n선택됨: {', '.join(selected)}")
            return selected

        # 쉼표, 공백 또는 연속 숫자 파싱
        chars = user_input.replace(',', ' ').replace(' ', '')
        selected = []
        invalid = []

        for char in chars:
            if char in available_formats:
                fmt_name = available_formats[char][0]
                if fmt_name not in selected:
                    selected.append(fmt_name)
            else:
                invalid.append(char)

        if invalid:
            print(f"잘못된 입력: {', '.join(invalid)}")
            continue

        if not selected:
            print("최소 하나 이상 선택해주세요.")
            continue

        print(f"\n선택됨: {', '.join(selected)}")
        confirm = input("계속하시겠습니까? (Y/n): ").strip().lower()

        if confirm in ['', 'y', 'yes']:
            return selected
        else:
            print("다시 선택해주세요.")


def sanitize_filename(text: str, max_length: int = 80) -> str:
    """
    파일명에 안전한 문자열로 변환
    - 특수문자 제거
    - 공백을 언더스코어로 변경
    - 최대 길이 제한
    """
    # 영문, 숫자, 한글, 공백, 하이픈, 언더스코어만 남기기
    text = re.sub(r'[^\w\s\-가-힣]', '', text)
    # 공백을 언더스코어로 변경
    text = re.sub(r'\s+', '_', text.strip())
    # 연속된 언더스코어 제거
    text = re.sub(r'_{2,}', '_', text)
    # 최대 길이 제한
    if len(text) > max_length:
        text = text[:max_length].rstrip('_')
    return text


def create_output_filename(result, source_name: str) -> str:
    """
    추출 결과로부터 출력 파일명 생성
    YouTube: {제목}_{채널명}_{공개날짜}
    Video: {제목}_{생성날짜}
    PDF: {제목}_{저자}_{생성날짜}
    Web: {제목}_{도메인}_{날짜}
    기타: {source_name}_{timestamp}
    """
    if source_name == 'youtube' and hasattr(result, 'title'):
        # YouTube 파일명: 제목_채널명_날짜
        title = sanitize_filename(result.title, max_length=30)
        channel = sanitize_filename(result.channel, max_length=20)
        upload_date = result.upload_date if result.upload_date else datetime.now().strftime('%Y%m%d')

        # 빈 값 처리
        if not title or title == 'YouTube_Video':
            title = 'youtube'
        if not channel or channel == 'Unknown':
            channel = 'unknown'

        return f"{title}_{channel}_{upload_date}"

    elif source_name == 'video' and hasattr(result, 'title'):
        # 로컬 비디오 파일명: 제목_날짜
        title = sanitize_filename(result.title, max_length=40)
        creation_date = result.creation_date if result.creation_date else datetime.now().strftime('%Y%m%d')

        # 빈 값 처리
        if not title:
            title = 'video'

        return f"{title}_{creation_date}"

    elif source_name == 'pdf' and hasattr(result, 'title'):
        # PDF 파일명: 제목_저자_날짜
        title = sanitize_filename(result.title, max_length=40)
        author = sanitize_filename(result.author, max_length=20) if result.author else ''
        creation_date = result.creation_date if result.creation_date else datetime.now().strftime('%Y%m%d')

        # 빈 값 처리
        if not title:
            title = 'pdf'

        if author:
            return f"{title}_{author}_{creation_date}"
        else:
            return f"{title}_{creation_date}"

    elif source_name == 'web' and hasattr(result, 'title'):
        # 웹페이지 파일명: 제목_도메인_날짜
        title = sanitize_filename(result.title, max_length=40)
        domain = sanitize_filename(result.domain.replace('.', '_'), max_length=20) if result.domain else ''
        # date 형식 정규화 (YYYY-MM-DD → YYYYMMDD)
        date_str = result.date.replace('-', '').replace('/', '')[:8] if result.date else datetime.now().strftime('%Y%m%d')

        # 빈 값 처리
        if not title:
            title = 'web'
        if not domain:
            domain = 'unknown'

        return f"{title}_{domain}_{date_str}"

    else:
        # 기본 파일명: source_timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{source_name}_{timestamp}"


def main():
    parser = argparse.ArgumentParser(
        description='콘텐츠 요약 노트 생성기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --youtube "URL"                    # 대화형 형식 선택 (기본)
  python main.py --video "./video.mp4"              # 대화형 형식 선택
  python main.py --pdf "./document.pdf" --formats all  # 전체 형식 자동
  python main.py --youtube "URL" -f detailed,easy   # 특정 형식만
  python main.py --web "URL" --extract-only         # 추출만 (노트 생성 안함)
        """
    )
    
    # 입력 소스 (상호 배타적)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--youtube', '-y', metavar='URL', help='YouTube URL')
    source_group.add_argument('--video', '-v', metavar='PATH', help='로컬 비디오 파일 경로')
    source_group.add_argument('--pdf', '-p', metavar='PATH', help='PDF 파일 경로')
    source_group.add_argument('--web', '-w', metavar='URL', help='웹페이지 URL')
    
    # 출력 옵션
    parser.add_argument('--output-dir', '-o', default='./output', help='출력 디렉토리 (기본: ./output)')
    parser.add_argument('--formats', '-f', default=None,
                       help='생성할 형식 (all, detailed, essence, easy, mindmap) 쉼표로 구분. 미지정시 대화형 선택')
    parser.add_argument('--lang', '-l', default='ko', help='출력 언어 (기본: ko)')
    
    # 추가 옵션
    parser.add_argument('--extract-only', action='store_true', help='추출만 수행 (노트 생성 안 함)')
    parser.add_argument('--generate-notes', '--auto', action='store_true',
                       help='노트 자동 생성 (ANTHROPIC_API_KEY 환경변수 필요)')
    parser.add_argument('--save-prompts', action='store_true',
                       help='프롬프트를 파일로 저장 (API 없이 사용)')
    parser.add_argument('--verbose', action='store_true', help='상세 출력')

    # 로컬 비디오 전용 옵션
    parser.add_argument('--model', '-m', default='medium',
                       choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
                       help='Whisper 모델 크기 (로컬 비디오 전용, 기본: medium)')

    # Vision 분석 옵션 (YouTube, 로컬 비디오)
    parser.add_argument('--with-vision', action='store_true',
                       help='화면 분석용 프레임 추출 (Claude Code Read 도구로 분석)')
    parser.add_argument('--vision-method', choices=['scene', 'interval'],
                       default='scene', help='프레임 추출 방식 (기본: scene - 장면변화 감지)')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='최대 프레임 수 (기본: 100)')

    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 형식 파싱
    if args.extract_only:
        # 추출만 하는 경우 형식 선택 불필요
        formats = []
    elif args.formats is None:
        # --formats 미지정 시 대화형 선택 (기본 동작)
        formats = select_formats_interactive()
        if not formats:
            print("형식이 선택되지 않았습니다. 종료합니다.")
            sys.exit(0)
    elif args.formats == 'all':
        formats = ['detailed', 'essence', 'easy', 'mindmap']
    else:
        formats = [f.strip() for f in args.formats.split(',')]
    
    print(f"\n{'='*50}")
    print("Content Summarizer")
    print(f"{'='*50}\n")
    
    # 1. 추출 단계
    print("[1/3] 콘텐츠 추출 중...")
    
    if args.youtube:
        from extractors.youtube import extract_youtube, to_json
        result = extract_youtube(
            args.youtube,
            args.lang,
            with_vision=args.with_vision,
            vision_method=args.vision_method,
            max_frames=args.max_frames
        )
        source_name = 'youtube'
    elif args.video:
        from extractors.video import extract_video, to_json
        result = extract_video(
            args.video,
            args.lang,
            args.model,
            with_vision=args.with_vision,
            vision_method=args.vision_method,
            max_frames=args.max_frames
        )
        source_name = 'video'
    elif args.pdf:
        from extractors.pdf import extract_pdf, to_json
        result = extract_pdf(args.pdf)
        source_name = 'pdf'
    elif args.web:
        from extractors.web import extract_web, to_json
        result = extract_web(args.web)
        source_name = 'web'
    
    if not result.success:
        print(f"\n[ERROR] 추출 실패: {result.warnings}")
        sys.exit(1)

    print(f"   [OK] 추출 완료 (품질: {result.quality_score}/100)")
    if result.warnings:
        print(f"   [WARN] 경고: {', '.join(result.warnings)}")

    # 출력 파일명 생성
    base_filename = create_output_filename(result, source_name)

    # 추출 결과 저장
    extract_path = os.path.join(args.output_dir, f'{base_filename}_extracted.json')
    with open(extract_path, 'w', encoding='utf-8') as f:
        f.write(to_json(result))
    print(f"   -> 추출 데이터: {extract_path}")

    if args.extract_only:
        print("\n추출 완료 (--extract-only 모드)")
        return

    # 2. 원문 저장
    print("\n[2/3] 원문 저장 중...")

    raw_path = os.path.join(args.output_dir, f'{base_filename}_raw.md')
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(f"# 원문 스크립트\n\n")
        f.write(f"- 출처: {getattr(result, 'source_url', '') or getattr(result, 'source_path', '')}\n")
        # YouTube/동영상인 경우 video_id 저장 (임베딩용)
        if hasattr(result, 'video_id') and result.video_id:
            f.write(f"- video_id: {result.video_id}\n")
        f.write(f"- 추출일: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- 품질점수: {result.quality_score}/100\n")

        # Vision 분석 정보
        if hasattr(result, 'frames_dir') and result.frames_dir:
            f.write(f"- 프레임 폴더: {result.frames_dir}\n")
            f.write(f"- 프레임 수: {len(result.frames) if result.frames else 0}개\n")

        f.write("\n---\n\n")
        f.write(result.full_text)

        # 프레임 정보 추가
        if hasattr(result, 'frames') and result.frames:
            f.write("\n\n---\n\n## 화면 캡처 정보\n\n")
            f.write("아래 프레임들을 Claude Code Read 도구로 분석할 수 있습니다:\n\n")
            for frame in result.frames:
                f.write(f"- [{frame['timestamp_str']}] {frame['path']}\n")

    print(f"   [OK] 원문 저장: {raw_path}")

    # Vision 프레임 정보 출력
    if hasattr(result, 'frames_dir') and result.frames_dir:
        print(f"   [OK] 프레임 폴더: {result.frames_dir}")
        print(f"   [OK] 추출된 프레임: {len(result.frames) if result.frames else 0}개")
    
    # 3. 노트 생성
    print("\n[3/3] 노트 생성...")
    print(f"\n[NOTE] 요청된 형식: {', '.join(formats)}")

    if args.generate_notes or args.save_prompts:
        from generators.note_generator import generate_all_notes

        # API 키 확인
        api_key = None
        if args.generate_notes:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print("\n[ERROR] ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
                print("   프롬프트만 생성합니다...\n")

        # 노트 생성 (검증 루프 포함)
        results = generate_all_notes(
            raw_path,
            args.output_dir,
            api_key,
            formats,
            use_verification=True,  # Level 1: Self-Critique
            source_type=source_name
        )

        # 결과 출력
        print(f"\n[DONE] 완료!")
        print(f"   - 추출 데이터: {extract_path}")
        print(f"   - 원문 파일: {raw_path}")

        success_count = sum(1 for r in results.values() if r.get('success'))
        print(f"   - 생성된 노트: {success_count}/{len(formats)}")

        for fmt, result in results.items():
            if result.get('success'):
                if 'note_path' in result:
                    print(f"     [OK] {fmt}: {result['note_path']}")
                elif 'prompt_path' in result:
                    print(f"     [OK] {fmt} 프롬프트: {result['prompt_path']}")

    else:
        # 수동 노트 생성 안내
        print("\n" + "="*50)
        print("노트 생성 방법")
        print("="*50)
        print("""
옵션 1: 자동 생성 (권장)
  python main.py --youtube "URL" --generate-notes
  (ANTHROPIC_API_KEY 환경변수 필요)

옵션 2: 프롬프트 생성
  python generators/note_generator.py {raw_path} --all --save-prompt

옵션 3: Claude Code에서 수동 생성
  "templates 폴더의 모든 템플릿을 참조하여
   {raw_path} 원문으로 4가지 노트를 모두 생성해줘"
""".format(raw_path=raw_path))

        print(f"\n[DONE] 추출 완료!")
        print(f"   - 추출 데이터: {extract_path}")
        print(f"   - 원문 파일: {raw_path}")


if __name__ == '__main__':
    main()
