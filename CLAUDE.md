# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Content Summarizer is an AI-powered system for extracting and summarizing content from YouTube videos, PDFs, and web pages. The system prioritizes accuracy by only including content from the source material and always preserving citations.

---

## Claude Code 워크플로우 (중요!)

사용자가 소스(YouTube URL, PDF, 동영상, 웹페이지)를 제공하면 다음 워크플로우를 따릅니다:

### 1단계: 노트 형식 및 생성 모드 선택 (AskUserQuestion 사용)
```
사용자가 소스를 입력하면 즉시 AskUserQuestion 도구를 사용하여 선택지를 물어봅니다:

질문 1: 노트 형식
- question: "어떤 노트 형식을 생성할까요?"
- header: "노트 형식"
- multiSelect: true
- options:
  1. Detailed - 상세 노트 (계층적 구조의 포괄적인 노트)
  2. Essence - 핵심 노트 (5~10개 핵심 포인트)
  3. Easy - 쉬운 노트 (초보자용 3~5개 핵심)
  4. Mindmap - 마인드맵 (Mermaid 다이어그램 + 트리 구조)

질문 2: 생성 모드
- question: "노트 생성 모드를 선택해주세요"
- header: "생성 모드"
- multiSelect: false
- options:
  1. Level 2 Agents (권장) - 분석→작성→검증 에이전트 파이프라인
  2. Simple - 단순 생성 (빠름, 검증 없음)
```

### 2단계: 콘텐츠 추출
```bash
python main.py --youtube "URL" -f [선택한형식] --extract-only
```

### 3단계: 노트 생성 (모드에 따라)

#### Simple 모드
- 템플릿 기반 직접 노트 생성
- 빠르지만 검증 없음

#### Level 2 Agents 모드 (권장)
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
YouTube나 동영상 소스인 경우, 생성된 노트 상단에 반드시 임베딩을 추가합니다:
```markdown
# [제목]

<iframe width="1280" height="720" src="https://www.youtube.com/embed/[VIDEO_ID]" frameborder="0" allowfullscreen></iframe>

## 메타 정보
...
```

### 워크플로우 예시 (Level 2 Agents 모드)
```
사용자: https://youtu.be/ABC123

Claude Code:
1. AskUserQuestion으로 노트 형식 + 생성 모드 선택
2. 사용자가 "Detailed" + "Level 2 Agents" 선택
3. python main.py --youtube "URL" --extract-only 실행
4. [Phase 1] Task 도구로 Analyst 에이전트 실행 → JSON 분석 결과 (언어 감지 포함)
5. [Phase 2] Task 도구로 Writer 에이전트 실행 → 노트 초안 (외국어면 번역 병기)
6. [Phase 3] Task 도구로 Critic 에이전트 실행 → 검증 점수
7. 검증 통과 시 노트 상단에 YouTube 임베딩 추가
8. output 폴더에 저장 및 완료 보고
```

### 언어 처리 워크플로우

외국어(영어, 일본어 등) 콘텐츠의 경우 자동으로 번역 병기 모드가 활성화됩니다:

```
┌─────────────────────────────────────────────────────────────────┐
│  Analyst Agent: source_language 감지                            │
│  - 원본 언어 식별 (ko, en, ja, zh, ...)                         │
│  - needs_translation 플래그 설정                                 │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Writer Agent: 조건부 번역 병기                                  │
│  - needs_translation=true → 원문 인용 + 한글 번역               │
│  - needs_translation=false → 일반 한국어 노트                   │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Critic Agent: 번역 품질 검증                                    │
│  - 원문 인용 포함 여부                                           │
│  - 번역 정확성                                                   │
│  - 용어 일관성                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**번역 병기 노트 예시 (영어 원본):**
```markdown
### 1. API Rate Limiting 이해

> "Rate limiting prevents API abuse by restricting the number of requests." [03:45]

**번역**: Rate limiting은 요청 수를 제한하여 API 남용을 방지합니다.

- **Rate Limiting (속도 제한)**: API 요청 빈도 제어 메커니즘
- **Throttling (스로틀링)**: 과도한 요청 시 속도 저하
```

---

## Level 2 에이전트 프롬프트 (Task 도구 사용)

### Analyst Agent 프롬프트
```
당신은 콘텐츠 분석 전문가입니다.
다음 원문을 분석하여 구조화된 JSON을 생성하세요.

## 분석 항목
1. main_topic: 주제
2. content_type: tutorial|lecture|interview|discussion|presentation
3. structure: [{section, timestamps, key_points}]
4. key_concepts: 핵심 개념 목록
5. difficulty_level: beginner|intermediate|advanced
6. source_language: 원본 언어 (ko|en|ja|zh|...)
7. needs_translation: 번역 필요 여부 (true: 한국어 아님, false: 한국어)

## 원문
[원문 텍스트]

## 출력 형식
반드시 JSON 형식으로 출력하세요.
```

### Writer Agent 프롬프트
```
당신은 노트 작성 전문가입니다.
분석 결과와 템플릿을 바탕으로 고품질 노트를 작성하세요.

## 핵심 규칙
1. 원문에 있는 내용만 포함 (환각 금지)
2. 모든 내용에 타임스탬프/페이지 인용 필수
3. 템플릿 형식 준수
4. 번역 병기: needs_translation=true인 경우 원문+한글번역 함께 작성

## 번역 병기 형식 (외국어 콘텐츠인 경우)
> "원문 인용" [타임스탬프]
**번역**: 한글 번역 내용
- 전문 용어는 `영어 (한글)` 형식으로 병기

## 분석 결과
[Analyst 출력 JSON]

## 원문
[원문 텍스트]

## 템플릿
[선택된 템플릿]
```

### Critic Agent 프롬프트
```
당신은 노트 품질 검증 전문가입니다.
생성된 노트가 원문에 충실하고 품질 기준을 충족하는지 검증합니다.

## 검증 기준
1. 원문 충실도: 노트의 모든 내용이 원문에 존재하는가?
2. 환각 탐지: 원문에 없는 내용이 추가되었는가?
3. 인용 정확성: 타임스탬프/페이지 번호가 정확한가?
4. 완성도: 핵심 내용이 누락되지 않았는가?
5. 번역 품질 (외국어인 경우): 원문 인용이 포함되어 있는가? 번역이 정확한가?

## 생성된 노트
[Writer 출력 노트]

## 원문
[원문 텍스트]

## 출력 형식
{
    "score": 0-100,
    "passed": true/false (80점 이상 통과),
    "issues": ["문제점 목록"],
    "suggestions": ["개선 제안"]
}
```

---

## Core Development Commands

### Installation
```bash
pip install -r requirements.txt
```

### Run Extractors
```bash
# YouTube extraction
python main.py --youtube "URL"

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
```

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
python extractors/pdf.py "./document.pdf"
python extractors/web.py "https://example.com/article"
```

## Architecture

### Two-Phase Design

The system operates in two distinct phases:

1. **Extraction Phase** (Python scripts)
   - Extracts raw content from sources
   - Validates extraction quality with scoring (0-100)
   - Preserves source location metadata (timestamps/page numbers/sections)
   - Outputs: JSON metadata + raw text file

2. **Note Generation Phase** (AI/manual)
   - Uses template prompts in `templates/` folder
   - Generates 4 note formats from extracted content
   - Must only use content from raw text (no inference)
   - Always includes source citations

### Extractor Pattern

All extractors (`extractors/youtube.py`, `extractors/pdf.py`, `extractors/web.py`) follow a common pattern:

1. Return `ExtractionResult` dataclass with:
   - `success`: bool
   - `source_type`: 'youtube'|'pdf'|'web'
   - `segments`: list with location metadata
   - `full_text`: formatted text with citations
   - `quality_score`: 0-100 with warnings
   - `warnings`: list of issues

2. Include `to_json()` function for serialization

3. Provide location mapping:
   - YouTube: timestamps `[HH:MM:SS]`
   - PDF: page numbers `[p.N]`
   - Web: section headings

### Main.py Workflow

1. Parse arguments (source type, output options)
2. Call appropriate extractor
3. Save JSON metadata: `{source}_{timestamp}_extracted.json`
4. Save raw text: `{source}_{timestamp}_raw.md`
5. Print instructions for manual note generation

Note: `main.py` does NOT generate notes automatically - it provides instructions for using the template prompts with AI tools.

## Note Generation Templates

Four templates in `templates/` directory define note formats:

- `detailed.md`: Comprehensive hierarchical notes (1. → 1.1 → 1.1.1)
- `essence.md`: 5-10 key points with concept relationships
- `easy.md`: 3-5 core points simplified for beginners
- `mindmap.md`: Mermaid diagram + text tree structure

Each template includes:
- Strict rules against inference/addition
- Required citation format
- Output structure specification
- DO/DON'T guidelines

## Critical Principles

When working with this codebase:

1. **Accuracy First**: Never add content not in the source
2. **Source Tracking**: All content must have location references
3. **Quality Validation**: Use quality_score and warnings systems
4. **Raw Preservation**: Always save complete raw text alongside summaries

## Automated Note Generation

The system now includes automated note generation:

- `generators/note_generator.py`: Core note generation module
  - Combines templates with extracted content
  - Supports Anthropic API for auto-generation
  - Falls back to prompt generation for manual use

- `summarize.py`: All-in-one wrapper script
  - Auto-detects source type (YouTube/PDF/Web)
  - Runs extraction + note generation in one command

- `quick_note.py`: Interactive note generator
  - Lists recent extractions
  - Dialog-based format selection
  - Quick regeneration of specific formats

Usage modes:
1. **Fully automated**: `--generate-notes` with ANTHROPIC_API_KEY
2. **Prompt generation**: `--save-prompts` for manual Claude.ai use
3. **Interactive**: `quick_note.py` for guided workflow

## Missing Components

The codebase does not include:
- Test suite (should test extractors, quality scoring, template parsing)
- Batch processing for multiple files
- Web UI or API server

If implementing these, maintain the two-phase separation and accuracy-first principles.
