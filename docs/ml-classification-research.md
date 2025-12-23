# ML-Based Classification Research

**Research Date**: 2025-12-23
**Status**: Technical Investigation Complete - Implementation Plan Ready

## Executive Summary

The music-organizer has an excellent foundation for ML integration with its rich domain model, high-performance string similarity (Rust extension), and extensible plugin system. This research identifies **7 high-value ML opportunities** ranging from quick wins (genre classification) to advanced features (acoustic similarity).

**Top Recommendations** (by effort/value ratio):
1. **Zero-dependency genre classifier** using scikit-learn (1 week effort)
2. **Content type classifier** with existing metadata (2 weeks effort)
3. **Audio feature extractor** using librosa (2 weeks effort, enables advanced features)

## Current Classification Architecture

### Existing Implementation

**File**: `src/music_organizer/core/classifier.py`

```python
# Current: Pattern-based classification with 5 content types
CONTENT_TYPES = {
    STUDIO: ["studio", "album", "original"],
    LIVE: ["live", "concert", "tour"],
    COMPILATION: ["compilation", "greatest", "best of"],
    COLLABORATION: ["feat", "&", "with", "versus"],
    RARITY: ["demo", "bootleg", "rare"]
}

# Limitations:
- Hard-coded patterns
- No learning from user corrections
- Limited to 5 content types
- Single-label classification only
```

**Domain Framework**: `src/music_organizer/domain/classification/`
- Rich domain model ready for ML integration
- Placeholder for `_extract_audio_features()` method
- Confidence scoring system (0.0-1.0)

## ML Opportunities

### 1. Genre Classification (Quick Win)

**Current**: Single genre from ID3 tags or basic heuristics
**ML Approach**: Multi-label genre classification using metadata

**Features**:
- Artist name (embedding)
- Album title (embedding)
- Track title (embedding)
- Year (temporal feature)
- Duration (numerical feature)

**Algorithm**: TfidfVectorizer + LogisticRegression
- Light weight (scikit-learn only)
- Fast inference (~1ms per track)
- Handles multi-label classification
- Minimal dependencies

**Expected Performance**:
- Accuracy: 70-80% on top 20 genres
- Inference time: <1ms per track
- Memory footprint: ~50MB model

**Implementation Effort**: 1 week

### 2. Enhanced Content Type Classification

**Current**: 5 types (studio, live, compilation, collaboration, rarity)
**ML Approach**: Add 5+ new types using metadata + audio features

**New Types**:
- REMIX (detected from title patterns + audio features)
- COVER (acoustic similarity to originals database)
- PODCAST (speech detection from audio features)
- SPOKEN_WORD (librosa speech detection)
- SOUNDTRACK (metadata patterns)
- BOOTLEG (quality metrics + metadata)

**Algorithm**: Random Forest Classifier
- Handles mixed feature types
- Good for small to medium datasets
- Interpretable feature importance

**Implementation Effort**: 2 weeks

### 3. Artist Collaboration Detection

**Current**: String matching for "feat", "&", "with"
**ML Approach**: Named Entity Recognition + relationship graph

**Features**:
- Artist name patterns
- Track position (lead vs. featured)
- Collaboration history
- Genre context

**Algorithm**: Rule-based + ML hybrid
- Spacy NER for artist names
- Custom heuristics for collaboration types
- Network analysis for frequent collaborators

**Implementation Effort**: 2 weeks

### 4. Audio Feature Extraction

**Current**: Basic metadata only (duration, file size, genre)
**ML Approach**: Extract psychoacoustic features using librosa

**Features**:
- Tempo (BPM)
- Key (chroma features)
- Energy (RMS energy)
- Danceability (rhythmic patterns)
- Valence (mood)
- Acousticness (spectral features)
- Speechiness (MFCC analysis)

**Library**: librosa (Python audio analysis)
- Industry standard for audio feature extraction
- Pure Python + numpy (no ML frameworks needed)
- Can extract features from all supported formats

**Implementation Effort**: 2 weeks (enables features 5-7)

### 5. Mood and Energy Classification

**Current**: Basic energy from duration/genre
**ML Approach**: Multi-dimensional mood classification

**Mood Dimensions**:
- Valence (positive/negative)
- Energy (high/low)
- Danceability (groove patterns)
- Acousticness (electronic vs organic)

**Algorithm**: SVM or Random Forest on audio features
- Requires audio feature extraction (#4)
- Can classify mood without metadata

**Use Cases**:
- Playlist generation (workout, focus, relaxation)
- Library organization by mood
- Music recommendation

**Implementation Effort**: 1 week (after #4)

### 6. Acoustic Similarity and Cover Detection

**Current**: File hash and metadata similarity
**ML Approach**: Chroma-based fingerprint comparison

**Algorithm**:
1. Extract chroma features (12-dimensional pitch class profile)
2. Cross-correlation for similarity
3. Dynamic time warping for tempo invariance

**Library**: librosa chroma CQT
- Detects covers and remixes
- Key-invariant comparison
- Tempo-invariant comparison

**Use Cases**:
- Cover song detection
- Duplicate detection across versions
- Version grouping (original vs. remix)

**Implementation Effort**: 2 weeks

### 7. Anomaly Detection

**Current**: None
**ML Approach**: Isolation Forest or One-Class SVM

**Use Cases**:
- Detect mislabeled genres
- Identify corrupt files
- Find outliers in collections
- Quality assessment

**Algorithm**: Isolation Forest
- Unsupervised learning
- No labeled data needed
- Fast inference

**Implementation Effort**: 1 week

## Proposed Implementation Roadmap

### Phase 1: Quick Wins (2-3 weeks)

**Task 1.1: Genre Classifier**
- Create `src/music_organizer/ml/genre_classifier.py`
- Use scikit-learn with TF-IDF features
- Train on MusicBrainz genre data (public dataset)
- Integrate with existing classifier plugin

**Task 1.2: Enhanced Content Types**
- Extend `ContentType` enum with new types
- Add REMIX, PODCAST, SPOKEN_WORD detection
- Use hybrid rule-based + ML approach

**Deliverables**:
- Working genre classifier
- New content types
- Tests and documentation
- No breaking changes to existing API

### Phase 2: Audio Features (2 weeks)

**Task 2.1: Librosa Integration**
- Create `src/music_organizer/ml/audio_features.py`
- Extract tempo, key, energy, valence
- Async feature extraction with progress tracking
- Cache features in SQLite database

**Task 2.2: Mood Classification**
- Train mood classifier on extracted features
- Create mood-based organization rules
- Add mood filtering to CLI

**Deliverables**:
- Audio feature extraction module
- Mood classifier
- Cached feature database
- CLI integration

### Phase 3: Advanced Features (2-3 weeks)

**Task 3.1: Acoustic Similarity**
- Implement chroma-based similarity
- Add cover song detection
- Integrate with duplicate detector

**Task 3.2: Anomaly Detection**
- Implement Isolation Forest
- Add quality assessment
- Create outlier reports

**Deliverables**:
- Acoustic similarity module
- Cover detection
- Anomaly detection
- Quality reports

## Technology Stack

### ML Libraries

| Library | Purpose | Size | License |
|---------|---------|------|--------|
| scikit-learn | Classic ML algorithms | ~50MB | BSD |
| librosa | Audio feature extraction | ~20MB | ISC |
| numpy | Numerical computing | ~20MB | BSD |
| scipy | Scientific computing | ~40MB | BSD |

**Optional / Future**:
- tensorflow/pytorch: For deep learning models
- transformers: For NLP-based metadata analysis
- spacy: For artist name NER

### Data Sources

**Training Data**:
- MusicBrainz dataset (public, CC0)
- Last.fm dataset (public API)
- Million Song Dataset (public, academic use)

**User Data**:
- Privacy-first: no data leaves user's machine
- Optional: Contribute anonymized data for model improvement
- Models ship with application (no external API calls)

## Architecture Design

### ML Module Structure

```
src/music_organizer/ml/
├── __init__.py
├── base.py              # Base ML model interface
├── genre_classifier.py   # Genre classification
├── content_classifier.py # Content type classification
├── audio_features.py     # Audio feature extraction
├── mood_classifier.py    # Mood and energy classification
├── acoustic_similarity.py# Acoustic similarity and cover detection
├── anomaly_detector.py   # Outlier and anomaly detection
└── models/              # Trained model files
    ├── genre_model.pkl
    ├── mood_model.pkl
    └── anomaly_model.pkl
```

### Integration Points

```python
# Extend existing classifier
class MLClassifierPlugin(ClassificationPlugin):
    async def classify(self, audio_file: AudioFile) -> Result[ContentType]:
        # 1. Try rule-based classification first
        result = self._rule_based_classify(audio_file)
        if result.confidence > 0.8:
            return result

        # 2. Fall back to ML classification
        return await self._ml_classify(audio_file)

# Extend metadata extraction
class MLEnhancedMetadataExtractor(MetadataExtractor):
    async def extract(self, file_path: Path) -> AudioFile:
        # 1. Extract basic metadata (mutagen)
        audio_file = await self._basic_extract(file_path)

        # 2. Extract audio features (librosa)
        features = await self._extract_audio_features(file_path)
        audio_file.audio_features = features

        # 3. Run ML classifiers
        audio_file.genre = await self.genre_classifier.predict(audio_file)
        audio_file.mood = await self.mood_classifier.predict(features)

        return audio_file
```

## Performance Considerations

### Feature Extraction Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Basic metadata | 10-50ms | Current (mutagen) |
| Audio feature extraction | 500-2000ms | librosa analysis |
| Genre classification | <1ms | scikit-learn inference |
| Mood classification | <1ms | scikit-learn inference |
| Acoustic similarity | 100-500ms | Chroma correlation |

### Optimization Strategies

1. **Cache audio features** in SQLite database (by file hash)
2. **Batch processing** for feature extraction (parallel workers)
3. **Lazy loading** - only extract features when needed
4. **Progressive enhancement** - use rule-based first, ML as fallback
5. **Background processing** - extract features during idle time

### Memory Usage

| Component | Memory |
|-----------|--------|
| scikit-learn models | ~50MB |
| librosa (loaded) | ~100MB |
| Audio buffer (one file) | ~50MB (for FLAC) |
| Feature cache | ~1MB per 1000 tracks |

## Privacy and Data Usage

### Privacy-First Design

1. **No external API calls** - All processing local
2. **No data collection** - User data stays on device
3. **Optional opt-in** - Users can contribute anonymized data
4. **Transparent models** - Open source models and training data

### Training Data Sources

**Public Domain / CC0**:
- MusicBrainz (genre metadata)
- Free Music Archive (audio features)
- Internet Archive (live recordings)

**User-Contributed (Optional)**:
- Anonymized metadata only (no audio files)
- Opt-in via CLI flag
- Used to improve model accuracy

## Cost-Benefit Analysis

### Benefits

| Feature | Value | Effort |
|---------|-------|--------|
| Genre classification | High | Low |
| Enhanced content types | Medium | Low |
| Audio features | High | Medium |
| Mood classification | Medium | Low (after features) |
| Acoustic similarity | Medium | High |
| Anomaly detection | Low | Low |

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Additional dependencies | Medium | Optional ML feature, core unaffected |
| Performance degradation | High | Lazy loading, caching, background processing |
| Privacy concerns | High | All processing local, opt-in only |
| Model accuracy | Medium | Fallback to rule-based, user correction feedback |

## Recommendations

### Immediate Actions (Next Sprint)

1. **Add optional ML dependencies** to pyproject.toml:
   ```toml
   [project.optional-dependencies]
   ml = ["scikit-learn>=1.3.0", "librosa>=0.10.0"]
   ```

2. **Create ML module structure**:
   - `src/music_organizer/ml/` directory
   - Base ML model interface
   - Plugin integration

3. **Implement genre classifier**:
   - Use MusicBrainz dataset for training
   - TF-IDF + LogisticRegression
   - Integrate with existing classifier

### Future Considerations

1. **Deep learning models** for better accuracy (larger dependencies)
2. **WebAssembly port** of genre classifier for browser-based preview
3. **Fine-tuning** on user data (with explicit consent)
4. **Cloud API** option for users who want cloud-based processing

## Conclusion

The music-organizer has an excellent foundation for ML integration. The **recommended approach** is:

1. **Start small** with genre classification (quick win, high value)
2. **Add audio features** as optional dependency (enables advanced features)
3. **Maintain backward compatibility** - ML enhances, doesn't replace existing features
4. **Privacy-first** - All processing local, no data collection

**Estimated total effort**: 6-8 weeks for full implementation
**Value**: Enhanced classification accuracy, new features (mood, acoustic similarity)

---

*Research by task-master agent on 2025-12-23*
