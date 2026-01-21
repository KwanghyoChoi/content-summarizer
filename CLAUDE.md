# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Content Summarizer is an AI-powered system for extracting and summarizing content from YouTube videos, local videos, PDFs, and web pages. The system prioritizes accuracy by only including content from the source material and always preserving citations.

---

## Claude Code 워크플로우 (중요!)

사용자가 소스(YouTube URL, 로컬 비디오, PDF, 웹페이지)를 제공하면 다음 워크플로우를 따릅니다:

### 1단계: 노트 형식 선택 (AskUserQuestion 사용)
```
사용자가 소스를 입력하면 즉시 AskUserQuestion 도구를 사용하여 노트 형식만 물어봅니다:

- question: "어떤 노트 형식을 생성할까요?"
- header: "노트 형식"
- multiSelect: true
- options:
  1. Detailed - 상세 노트 (계층적 구조의 포괄적인 노트)
  2. Essence - 핵심 노트 (5~10개 핵심 포인트)
  3. Easy - 쉬운 노트 (초보자용 3~5개 핵심)
  4. Mindmap - 마인드맵 (Mermaid 다이어그램 + 트리 구조)

※ 생성 모드는 항상 Level 2 Agents 사용 (질문하지 않음)
```

### 2단계: 콘텐츠 추출
```bash
python main.py --youtube "URL" -f [선택한형식] --extract-only
# 또는 로컬 비디오:
python main.py --video "./video.mp4" --extract-only
```

### 3단계: 노트 생성 (Level 2 Agents)

**Claude Code 내에서 Task 도구를 사용한 서브에이전트 실행**

```
┌─────────────────────────────────────────────────────────────────┐
│               Phase 1: Analyst Agent (분석가)                    │
│    Task 도구로 실행 - 콘텐츠 구조 분석 및 핵심 개념 추출           │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               Phase 2: Writer Agent (작성자)                     │
│    Task 도구로 실행 - 분석 결과 기반 노트 초안 작성               │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               Phase 3: Critic Agent (검증자)                     │
│    Task 도구로 실행 - 품질 검증 및 점수화 (80점 이상 통과)        │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
                     [검증 통과?]
                      ↙       ↘
                    Yes        No
                    ↓          ↓
                 [완료]    [수정 후 재검증]
```

### 4단계: YouTube/동영상 임베딩
YouTube나 동영상 소스인 경우, 생성된 노트 상단에 반드시 반응형 임베딩을 추가합니다:
```markdown
# [제목]

<!-- 반응형 YouTube 임베딩 (16:9 비율 유지, 창 크기에 맞춤) -->
<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin: 20px 0;">
  <iframe
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
    src="https://www.youtube.com/embed/[VIDEO_ID]"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

## 메타 정보
...
```

### 워크플로우 예시
```
사용자: https://youtu.be/ABC123

Claude Code:
1. AskUserQuestion으로 노트 형식 선택
2. 사용자가 "Detailed" 선택
3. python main.py --youtube "URL" --extract-only 실행
4. [Phase 1] Task 도구로 Analyst 에이전트 실행 → JSON 분석 결과
5. [Phase 2] Task 도구로 Writer 에이전트 실행 → 노트 초안
6. [Phase 3] Task 도구로 Critic 에이전트 실행 → 검증 점수
7. 검증 통과 시 노트 상단에 YouTube 임베딩 추가
8. output 폴더에 저장 및 완료 보고
```

---

## Level 2 에이전트 프롬프트 (Task 도구 사용)

에이전트 프롬프트 템플릿은 `templates/agents/` 폴더에 있습니다:
- `analyst_prompt.md` - 콘텐츠 구조 분석
- `writer_prompt.md` - 노트 작성
- `critic_prompt.md` - 품질 검증

### Analyst Agent 핵심 출력
```json
{
  "main_topic": "주제",
  "content_type": "tutorial|lecture|interview|discussion|presentation",
  "structure": [{"section": "섹션명", "timestamps": ["00:00-02:30"], "key_points": [...]}],
  "key_concepts": ["개념1", "개념2"],
  "difficulty_level": "beginner|intermediate|advanced",
  "channel_author": "채널명 (옵시디언 링크용)",
  "tags": ["#생산성", "#AI"],
  "related_concepts": ["[[제텔카스텐]]", "[[PKM]]"]
}
```

### Writer Agent 핵심 규칙
1. 원문에 있는 내용만 포함 (환각 금지)
2. 모든 내용에 타임스탬프/페이지 인용 필수
3. 옵시디언 링크: `[[채널명]]`, `#태그`, `[[관련개념]]`

### Critic Agent 출력
```json
{
  "score": 0-100,
  "passed": true/false,  // 80점 이상 통과
  "issues": ["문제점"],
  "suggestions": ["개선 제안"]
}
```

---

## Core Development Commands

### Installation
```bash
pip install -r requirements.txt

# For local video transcription (Whisper)
pip install openai-whisper opencv-python
```

### Run Extractors
```bash
# YouTube extraction
python main.py --youtube "URL"

# Local video extraction (uses Whisper for transcription)
python main.py --video "./video.mp4"

# PDF extraction
python main.py --pdf "path/to/file.pdf"

# Web page extraction
python main.py --web "URL"

# Extract only (skip note generation)
python main.py --youtube "URL" --extract-only

# Extract and auto-generate notes (requires ANTHROPIC_API_KEY)
python main.py --youtube "URL" --generate-notes

# Specify output formats
python main.py --youtube "URL" --formats detailed,essence

# With vision analysis (screen capture + audio)
python main.py --video "./video.mp4" --with-vision
python main.py --youtube "URL" --with-vision --max-frames 50
```

### Vision Analysis (화면 분석)

`--with-vision` 옵션은 음성 인식과 함께 화면 캡처를 수행합니다:

```bash
# 기본 사용 (장면 변화 감지)
python main.py --video "./video.mp4" --with-vision

# 고정 간격으로 캡처 (30초마다)
python main.py --video "./video.mp4" --with-vision --vision-method interval

# 최대 프레임 수 제한
python main.py --youtube "URL" --with-vision --max-frames 50
```

**작동 방식:**
1. 프레임 추출 (`extractors/frames.py`)
   - `scene`: 장면 변화 감지 (기본값, 효율적)
   - `interval`: 고정 간격 추출
2. 프레임을 이미지 파일로 저장
3. Claude Code의 Read 도구로 직접 분석 (추가 비용 없음)

**출력:**
- `{video}_frames/` 폴더에 프레임 저장
- `_raw.md` 파일에 프레임 경로 목록 포함

**사용 시점:**
- 코딩 튜토리얼 (화면 코드 캡처)
- 슬라이드 프레젠테이션
- 음성으로 설명하지 않은 시각적 정보가 있을 때

### Quick Start (All-in-One)
```bash
# Simplest way - auto-detect source type
python summarize.py "URL or file path" --auto

# Examples
python summarize.py "https://youtube.com/watch?v=..." --auto
python summarize.py "./document.pdf" --auto
python summarize.py "https://example.com/article" --formats essence,mindmap --auto
```

### Generate Notes from Extracted Content
```bash
# Generate all 4 note formats
python generators/note_generator.py output/youtube_*_raw.md --all --auto

# Generate specific format
python generators/note_generator.py output/youtube_*_raw.md --format detailed --auto

# Interactive mode
python quick_note.py

# Generate prompts only (for manual use with Claude.ai)
python generators/note_generator.py output/youtube_*_raw.md --all --save-prompt
```

### Run Individual Extractors
```bash
# Test extractors directly
python extractors/youtube.py "https://youtube.com/watch?v=..."
python extractors/video.py "./video.mp4"
python extractors/pdf.py "./document.pdf"
python extractors/web.py "https://example.com/article"
```

## Architecture

### Two-Phase Design

```
┌──────────────────────────────────────────────────────────────┐
│  Phase 1: Extraction (Python)                                │
│  extractors/*.py → ExtractionResult → JSON + raw.md          │
└──────────────────────────────┬───────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────┐
│  Phase 2: Note Generation (AI)                               │
│  templates/*.md + raw.md → Anthropic API → notes.md          │
└──────────────────────────────────────────────────────────────┘
```

### Extractor Pattern

All extractors (`extractors/youtube.py`, `video.py`, `pdf.py`, `web.py`) return `ExtractionResult`:

```python
@dataclass
class ExtractionResult:
    success: bool
    source_type: str      # 'youtube'|'video'|'pdf'|'web'
    segments: list        # [{start, end, text}] or [{page, text}]
    full_text: str        # Formatted with citations
    quality_score: int    # 0-100
    warnings: list        # Quality issues
```

Location mapping by source type:
- YouTube/Video: `[HH:MM:SS]` timestamps
- PDF: `[p.N]` page numbers
- Web: `## Section` headings

### Multi-Agent Architecture

**Claude Code Task Agents (기본 사용):**
- Claude Code의 Task 도구로 서브에이전트 실행
- Analyst→Writer→Critic 파이프라인
- 프롬프트: `templates/agents/*.md`
- 80점 이상 통과 시 완료, 미달 시 수정 후 재검증

**Python-based Agents (`generators/agents/`) - 대체 옵션:**
- `--generate-notes` 플래그 + ANTHROPIC_API_KEY로 사용
- 동일한 파이프라인을 Python으로 구현

### Output File Naming

```
{title}_{channel/author}_{date}_extracted.json  # Metadata
{title}_{channel/author}_{date}_raw.md          # Full text
{title}_{channel/author}_{date}_detailed.md     # Note format
```

## Note Generation Templates

Four templates in `templates/`:

| Template | Purpose | Structure | Length |
|----------|---------|-----------|--------|
| `detailed.md` | Full content | 1. → 1.1 → 1.1.1 | 80-100% |
| `essence.md` | Key concepts | 5-10 points | 30-40% |
| `easy.md` | Beginner-friendly | 3-5 core points | 10-20% |
| `mindmap.md` | Visual structure | Mermaid + tree | Keywords only |

## Critical Principles

1. **Accuracy First**: Never add content not in the source
2. **Source Tracking**: All content must have location references
3. **Quality Validation**: Use quality_score (0-100) and warnings
4. **Raw Preservation**: Always save complete raw text alongside summaries

## Entry Points

| Script | Purpose |
|--------|---------|
| `main.py` | Primary CLI with extraction + optional note generation |
| `summarize.py` | Auto-detects source type, runs full pipeline |
| `quick_note.py` | Interactive note regeneration from existing extractions |
| `generators/note_generator.py` | Direct note generation from raw files |
