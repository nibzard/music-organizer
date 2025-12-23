# WebAssembly Investigation: Browser-Based Music Organization

**Investigation Date**: 2025-12-23
**Status**: Feasibility Study - Hybrid Architecture Recommended

## Executive Summary

After analyzing the music-organizer codebase, **pure browser-based organization is not feasible** due to fundamental browser limitations (file system access, mutagen dependency). However, a **hybrid architecture** using WebAssembly for specific compute-intensive tasks is technically viable and could provide value for specific use cases.

**Recommendation**: Pursue WebAssembly for **client-side classification and metadata preview** only, not full file organization.

## Technical Constraints

### Browser Limitations

| Constraint | Impact | Workaround |
|------------|--------|------------|
| No direct file system access | Cannot scan/move/rename files | File API (user-selected files only) |
| No directory traversal | Cannot recursively scan libraries | Manual folder selection (limited) |
| Memory limits (~500MB-2GB) | Cannot process large libraries | Chunked processing |
| No mutagen in browser | Cannot extract audio metadata | Server-side processing required |
| No subprocess execution | Cannot run external tools | N/A |

### Codebase Analysis

**WASM-Compatible Modules** (pure Python logic):
- `src/music_organizer/core/classifier.py` - Content classification
- `src/music_organizer/utils/string_similarity.py` - String algorithms
- `src/music_organizer/models/audio_file.py` - Data models

**WASM-Incompatible Modules** (system dependencies):
- `src/music_organizer/core/metadata.py` - Requires mutagen
- `src/music_organizer/core/mover.py` - File system operations
- `src/music_organizer/core/async_file_mover.py` - Async file operations

## Proposed Hybrid Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Browser (WASM)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ String       │  │ Content      │  │ Pattern      │      │
│  │ Similarity   │  │ Classifier   │  │ Matcher      │      │
│  │ (Levenshtein │  │ (Live/Comp/  │  │ (Regex rules │      │
│  │  Jaro-Winkler│  │  Collab det.) │  │  engine)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Server (Python)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Metadata     │  │ File         │  │ Organization │      │
│  │ Extraction   │  │ Operations   │  │ Engine       │      │
│  │ (mutagen)    │  │ (scan/move)  │  │ (full logic) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Strategy

#### Phase 1: WASM String Similarity Module
**Effort**: Low | **Value**: Medium | **Risk**: Low

Compile `string_similarity.py` to WASM using Pyodide:
- Levenshtein, Jaro-Winkler, Jaccard algorithms
- Music metadata normalization
- Client-side duplicate detection for uploaded files

**Use Case**: Users upload track lists, browser identifies potential duplicates server-side.

#### Phase 2: WASM Classification Engine
**Effort**: Medium | **Value**: Medium | **Risk**: Medium

Port classification logic to WASM:
- Content type detection (studio/live/compilation)
- Artist collaboration detection
- Genre inference from metadata patterns

**Use Case**: Preview how files will be organized before uploading to server.

#### Phase 3: Web UI + File API Integration
**Effort**: High | **Value**: Low-Medium | **Risk**: High

Build browser interface for:
- File/folder selection via File API
- Client-side metadata preview
- Organization preview before server processing

**Limitation**: Cannot move actual files, only preview organization.

## Alternative: Server-Side Processing

Given the constraints, **a progressive web app (PWA) with server-side processing** is more practical:

### Recommended Architecture

```
┌─────────────────────┐
│   Web UI (React)    │
│  - File selection   │
│  - Preview          │
│  - Progress         │
└──────────┬──────────┘
           │ REST API
           ▼
┌─────────────────────┐
│   FastAPI Server    │
│  - Metadata extract │
│  - Classification   │
│  - File operations  │
└─────────────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | React + TypeScript + Vite |
| Backend | FastAPI (Python) |
| WASM | Pyodide (optional, for string ops) |
| Deployment | Docker + Nginx |

## Proof of Concept: Pyodide String Similarity

```python
# browser_demo.html
<script src="https://cdn.jsdelivr.net/pyodide/v0.24.0/full/pyodide.js"></script>
<script>
  async function loadStringSimilarity() {
    const pyodide = await loadPyodide();

    // Load string similarity module
    await pyodide.loadPackage(["micropip"]);
    await pyodide.runPythonAsync(`
      import micropip
      await micropip.install('music-organizer')
      from music_organizer.utils.string_similarity import music_metadata_similarity
    `);

    // Calculate similarity in browser
    const result = pyodide.runPython(
      `music_metadata_similarity("The Beatles", "Beatles, The")`
    );

    console.log('Similarity:', result);  // 1.0
  }
</script>
```

## Cost-Benefit Analysis

### WASM Approach Benefits
- Client-side processing reduces server load
- Real-time feedback for user interactions
- Offline-capable for certain operations

### WASM Approach Drawbacks
- Limited to metadata preview (cannot move files)
- Pyodide bundle size: ~10MB initial load
- Browser memory constraints
- Complex build toolchain
- Mutagen unavailable (server still needed)

### Server-Side-Only Benefits
- Simpler architecture
- Full functionality
- No browser memory limits
- Easier debugging

### Server-Side-Only Drawbacks
- Server required for all operations
- Network latency
- No offline capability

## Recommendation

**Do NOT pursue pure WASM browser-based organization.**

**Instead**: Build a **FastAPI web service** with optional WASM for:
1. Client-side duplicate detection in file selection UI
2. Real-time organization preview
3. String similarity matching for metadata correction

**Implementation Priority**:
1. Build REST API for existing organizer functionality
2. Create React web UI for file upload and organization preview
3. Optional: Add Pyodide for client-side string operations
4. Deploy as containerized web service

## Tools for Web/WASM Implementation

| Tool | Purpose |
|------|---------|
| Pyodide | Python runtime in WebAssembly |
| FastAPI | Python web framework |
| React + TypeScript | Frontend framework |
| Vite | Fast frontend build tool |
| nginx | Reverse proxy for production |

## Estimated Effort

| Task | Effort |
|------|--------|
| REST API wrapper for organizer | 1-2 weeks |
| React web UI | 2-3 weeks |
| Pyodide integration (optional) | 1-2 weeks |
| Deployment setup | 1 week |
| **Total** | **5-8 weeks** |

## Conclusion

WebAssembly is **not suitable** for core music organization functionality due to browser file system limitations. The recommended approach is a **web service with optional WASM enhancements** for preview operations.

**Next Steps**:
1. Build FastAPI wrapper for existing organizer
2. Create web UI for library management
3. Evaluate Pyodide for client-side preview features

---

*Investigation by task-master agent on 2025-12-23*
