# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Japanese literary text processing pipeline that analyzes "吾輩は猫である" (I Am a Cat) by Natsume Sōseki. The system performs:

1. **Text chunking** with character detection (rule-based + optional LLM)
2. **Event extraction** using LLM (Map-Reduce pattern with embedding-based clustering)
3. **Character persona timeline** generation across story progression
4. **Special rules** generation for banned phrases and style overrides

All scripts are independent and run sequentially to produce JSONL/JSON data files.

## Commands

### Setup

```bash
# Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Pipeline Execution

The scripts must be run in this order:

```bash
# 1. Generate chunks from main.txt (600-1400 chars each)
python app/make_chunks.py

# With LLM-assisted character detection:
python app/make_chunks.py --llm-ner --llm-ner-limit 100 --ner-min-heuristic 2

# 2. Extract events from chunks using Map-Reduce
python app/make_events.py

# 3. Generate character persona timelines
python app/make_persona_timeline.py

# 4. Create special rules (banned phrases, fixed replies, style overrides)
python app/make_special_rules.py
```

### Environment Variables

Set in `.env` file:

- `OPENAI_API_KEY`: Required for all LLM operations
- `CHAT_MODEL`: Defaults to `gpt-4o-mini`
- `EMBED_MODEL`: Defaults to `text-embedding-3-small`

## Architecture

### Data Flow

```
data/main.txt (input novel text)
    ↓
[make_chunks.py] → data/chunks.jsonl + data/ner_cache.json
    ↓
[make_events.py] → data/events.jsonl
    ↓
[make_persona_timeline.py] → data/character.json
    ↓
[make_special_rules.py] → data/special_rules.json
```

### make_chunks.py

**Purpose**: Splits the novel into semantic chunks (scenes) with character detection.

**Key features**:

- Chapter detection via regex (`第X章`)
- Sentence-based packing (target: 1000 chars, min: 600, max: 1400)
- Two-stage character detection:
  1. Rule-based: Uses `ALLOWED_NAMES` (64 canonical characters) + `ALIASES` dictionary for variant spellings
  2. Optional LLM fallback: When heuristic detection finds < `ner_min_heuristic` characters
- Caching: `ner_cache.json` stores LLM results by scene_id

**Output schema** (`chunks.jsonl`):

```json
{
  "id": "scene_00001",
  "chapter": 1,
  "scene_index": 1,
  "start_pos": 0,
  "end_pos": 1024,
  "characters": ["吾輩", "珍野 苦沙弥"],
  "text": "..."
}
```

### make_events.py

**Purpose**: Extract story events using Map-Reduce pattern with embedding-based deduplication.

**Pipeline**:

1. **Map phase**: Each chunk → LLM extracts 0+ events (type, title, importance 1-3)
2. **Reduce phase**:
   - Embed each event as `"{type}:{title}"`
   - Cluster by cosine similarity (threshold: 0.86)
   - Select representative (highest importance, earliest scene)
3. **Spoiler scoring**: Heuristic based on importance + story position

**Output schema** (`events.jsonl`):

```json
{
  "scene": 42,
  "type": "邂逅",
  "title": "迷亭との出会い",
  "spoiler_level": 2
}
```

### make_persona_timeline.py

**Purpose**: Capture character voice evolution over the story.

**Process**:

1. Identify top 8 frequent characters (≥2 chars, ≤10 chars, excluding noise words)
2. For each character, sample 6 representative chunks (start, middle, end + random)
3. LLM infers 2-phase timeline with style parameters:
   - `first_person`: Pronoun usage
   - `tone`: Speech style (皮肉屋, 内省的, etc.)
   - `tempo`: Pacing (ゆっくり, 早口, etc.)
   - `quirks`: Catchphrases

**Output schema** (`character.json`):

```json
{
  "character": "吾輩",
  "from_scene": 1,
  "to_scene": 120,
  "style": {
    "first_person": "吾輩",
    "tone": "皮肉屋",
    "tempo": "ゆっくり",
    "quirks": ["〜である"]
  }
}
```

### make_special_rules.py

**Purpose**: Generate conversation guard rails.

**Components**:

1. **banned_phrases**: Base list + auto-generated from event titles (e.g., "犯人は", "{event_word}は")
2. **fixed_replies**: Regex-based canned responses (e.g., gratitude → "礼には及ばない")
3. **style_overrides**: Context-based style adjustments (e.g., "！" → stronger tone)

**Output**: `data/special_rules.json`

## Key Design Patterns

### Character Detection Strategy

- **Allowlist-first**: Only canonical names in `ALLOWED_NAMES` can appear
- **Fuzzy matching**: Uses word boundary regex tailored for Japanese text
- **LLM as fallback**: Avoids hallucination by constraining to allowlist
- **Caching**: Prevents redundant API calls across reruns

### Event Clustering

- Embedding-based deduplication handles paraphrasing (e.g., "迷亭登場" ≈ "迷亭との出会い")
- Cluster representative selection prioritizes importance + chronology
- Reduces ~300 raw events → ~50-100 canonical events

### Persona Timeline

- Two-phase model balances simplicity vs. character arc capture
- Sampling strategy ensures early/late story coverage
- Fallback to neutral defaults if LLM output is malformed

## Important Constants

### make_chunks.py

- `TARGET_CHUNK = 1000`: Ideal chunk size in characters
- `MIN_CHUNK = 600`: Minimum before merging with next
- `MAX_CHUNK = 1400`: Hard limit before forced split
- Character limit per chunk: 8 (line 337)

### make_events.py

- Clustering threshold: `0.86` (line 118)
- Spoiler boost at >66% story position (lines 85-86)

### make_persona_timeline.py

- `SAMPLES_PER_CHAR = 6`: Chunks sampled per character
- Top 8 characters only (line 31)
