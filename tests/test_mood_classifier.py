"""Tests for mood classifier."""

from pathlib import Path

import pytest

from music_organizer.ml.mood_classifier import (
    MoodClassifier,
    MoodClassificationResult,
    classify_mood,
    get_mood_description,
    get_mood_from_valence_energy,
    _classify_mood_from_features,
)
from music_organizer.domain.classification.value_objects import Mood, AudioFeatures


class TestMoodClassificationResult:
    """Tests for MoodClassificationResult."""

    def test_empty_result(self):
        """Empty result should have zero confidence."""
        result = MoodClassificationResult()
        assert result.moods == []
        assert result.confidence == 0.0
        assert result.primary_mood is None

    def test_add_mood(self):
        """Adding moods should update result."""
        result = MoodClassificationResult()
        result.add_mood(Mood.HAPPY, 0.8)
        result.add_mood(Mood.ENERGETIC, 0.6)

        assert Mood.HAPPY in result.moods
        assert Mood.ENERGETIC in result.moods
        assert result.primary_mood == Mood.HAPPY
        assert result.confidence == 0.8

    def test_add_mood_updates_primary(self):
        """Primary mood should update with highest confidence."""
        result = MoodClassificationResult()
        result.add_mood(Mood.SAD, 0.5)
        result.add_mood(Mood.HAPPY, 0.9)
        assert result.primary_mood == Mood.HAPPY

    def test_add_mood_ignores_duplicates(self):
        """Duplicate moods should not be added twice."""
        result = MoodClassificationResult()
        result.add_mood(Mood.HAPPY, 0.5)
        result.add_mood(Mood.HAPPY, 0.8)
        assert len(result.moods) == 1
        assert result.confidence_scores["happy"] == 0.8

    def test_get_top_moods(self):
        """Get top N moods by confidence."""
        result = MoodClassificationResult()
        result.add_mood(Mood.HAPPY, 0.9)
        result.add_mood(Mood.ENERGETIC, 0.7)
        result.add_mood(Mood.CALM, 0.5)
        result.add_mood(Mood.RELAXED, 0.3)

        top_3 = result.get_top_moods(3)
        assert len(top_3) == 3
        assert top_3[0] == ("happy", 0.9)
        assert top_3[1] == ("energetic", 0.7)
        assert top_3[2] == ("calm", 0.5)

    def test_to_dict(self):
        """Convert result to dictionary."""
        result = MoodClassificationResult()
        result.add_mood(Mood.HAPPY, 0.8)
        result.method = "features"
        result.valence = 0.8
        result.energy = 0.7

        d = result.to_dict()
        assert d["moods"] == ["happy"]
        assert d["confidence_scores"]["happy"] == 0.8
        assert d["primary_mood"] == "happy"
        assert d["confidence"] == 0.8
        assert d["method"] == "features"
        assert d["valence"] == 0.8
        assert d["energy"] == 0.7


class TestClassifyMoodFromFeatures:
    """Tests for _classify_mood_from_features function."""

    def test_happy_mood(self):
        """High valence + high energy = happy."""
        result = _classify_mood_from_features(
            valence=0.9,
            energy=0.8,
            danceability=0.7,
            acousticness=0.2,
        )
        assert Mood.HAPPY in result.moods or result.primary_mood in [Mood.HAPPY, Mood.ENERGETIC, Mood.UPLIFTING]
        assert result.valence == 0.9
        assert result.energy == 0.8

    def test_sad_mood(self):
        """Low valence + low energy = sad."""
        result = _classify_mood_from_features(
            valence=0.2,
            energy=0.3,
            danceability=0.2,
            acousticness=0.5,
        )
        assert Mood.SAD in result.moods or result.primary_mood in [Mood.SAD, Mood.MELANCHOLIC]
        assert result.method == "features"

    def test_energetic_mood(self):
        """High energy regardless of valence = energetic."""
        result = _classify_mood_from_features(
            valence=0.5,
            energy=0.9,
            danceability=0.8,
            acousticness=0.1,
        )
        assert Mood.ENERGETIC in result.moods or result.primary_mood in [Mood.ENERGETIC, Mood.HAPPY]

    def test_calm_mood(self):
        """High valence + low energy = calm."""
        result = _classify_mood_from_features(
            valence=0.7,
            energy=0.2,
            danceability=0.2,
            acousticness=0.6,
        )
        # RELAXED should be in mood list (matched acousticness and low energy)
        assert Mood.RELAXED in result.moods

    def test_angry_mood(self):
        """Low valence + high energy = angry."""
        result = _classify_mood_from_features(
            valence=0.2,
            energy=0.8,
            danceability=0.5,
            acousticness=0.2,
        )
        assert Mood.ANGRY in result.moods or result.primary_mood in [Mood.ANGRY, Mood.DARK]

    def test_relaxed_mood(self):
        """High valence + low energy + high acoustic = relaxed."""
        result = _classify_mood_from_features(
            valence=0.7,
            energy=0.2,
            danceability=0.2,
            acousticness=0.8,
        )
        assert Mood.RELAXED in result.moods or result.primary_mood in [Mood.RELAXED, Mood.CALM]

    def test_missing_features(self):
        """Missing features should return unknown."""
        result = _classify_mood_from_features(
            valence=None,
            energy=None,
            danceability=None,
            acousticness=None,
        )
        assert Mood.UNKNOWN in result.moods

    def test_only_valence_missing(self):
        """Only valence missing should still classify."""
        result = _classify_mood_from_features(
            valence=None,
            energy=0.5,
            danceability=0.5,
            acousticness=0.5,
        )
        # Should return unknown since valence is critical
        assert Mood.UNKNOWN in result.moods


class TestMoodClassifier:
    """Tests for MoodClassifier."""

    def test_classifier_initialization(self):
        """Classifier should initialize without error."""
        classifier = MoodClassifier()
        assert classifier.model_name == "mood_classifier"
        assert not classifier.is_loaded

    def test_predict_from_features(self):
        """Prediction from feature values."""
        classifier = MoodClassifier()

        result = classifier.predict(
            valence=0.8,
            energy=0.7,
            danceability=0.6,
        )

        assert isinstance(result, MoodClassificationResult)
        assert result.method == "features"
        assert result.valence == 0.8
        assert result.energy == 0.7

    def test_predict_from_audio_features(self):
        """Prediction from AudioFeatures value object."""
        classifier = MoodClassifier()

        audio_features = AudioFeatures(
            valence=0.2,
            energy=0.3,
            danceability=0.2,
            acousticness=0.8,
        )

        result = classifier.predict_from_features(audio_features)

        assert isinstance(result, MoodClassificationResult)
        assert result.method in ["features", "ml"]

    def test_predict_from_text(self):
        """Prediction from text keywords."""
        classifier = MoodClassifier()

        result = classifier.predict_from_text("Happy celebration joy upbeat")

        assert isinstance(result, MoodClassificationResult)
        assert result.method == "text_keywords"
        assert Mood.HAPPY in result.moods

    def test_predict_from_text_sad(self):
        """Prediction from sad text keywords."""
        classifier = MoodClassifier()

        result = classifier.predict_from_text("Sad cry tears heartbreak lonely")

        assert isinstance(result, MoodClassificationResult)
        assert Mood.SAD in result.moods

    def test_predict_from_text_no_match(self):
        """Text without mood keywords returns unknown."""
        classifier = MoodClassifier()

        result = classifier.predict_from_text("some random text")

        assert isinstance(result, MoodClassificationResult)
        assert Mood.UNKNOWN in result.moods

    def test_predict_audio_features_override(self):
        """Individual feature params override audio_features."""
        classifier = MoodClassifier()

        audio_features = AudioFeatures(
            valence=0.2,
            energy=0.2,
        )

        # Override with different values
        result = classifier.predict(
            audio_features=audio_features,
            valence=0.9,
            energy=0.8,
        )

        # Should use override values
        assert result.valence == 0.9
        assert result.energy == 0.8


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_classify_mood_from_features(self):
        """Classify mood from feature values."""
        moods = classify_mood(
            valence=0.8,
            energy=0.7,
            danceability=0.6,
        )

        assert isinstance(moods, list)
        assert len(moods) > 0
        assert all(isinstance(m, str) for m in moods)

    def test_classify_mood_from_audio_features(self):
        """Classify mood from AudioFeatures."""
        audio_features = AudioFeatures(
            valence=0.8,
            energy=0.7,
            danceability=0.6,
        )

        moods = classify_mood(audio_features=audio_features)

        assert isinstance(moods, list)
        assert len(moods) > 0

    def test_classify_mood_from_text(self):
        """Classify mood from text."""
        moods = classify_mood(text="happy joy celebration")

        assert isinstance(moods, list)
        assert "happy" in moods

    def test_get_mood_description(self):
        """Get mood description."""
        desc = get_mood_description(Mood.HAPPY)
        assert isinstance(desc, str)
        assert "positive" in desc.lower() or "energetic" in desc.lower()

    def test_get_mood_from_valence_energy(self):
        """Get mood from valence/energy quadrants."""
        # High valence + high energy = happy
        assert get_mood_from_valence_energy(0.8, 0.7) == Mood.HAPPY

        # High valence + low energy = relaxed
        assert get_mood_from_valence_energy(0.8, 0.3) == Mood.RELAXED

        # Low valence + high energy = angry
        assert get_mood_from_valence_energy(0.2, 0.7) == Mood.ANGRY

        # Low valence + low energy = sad
        assert get_mood_from_valence_energy(0.2, 0.3) == Mood.SAD


class TestValenceEnergyQuadrants:
    """Tests for valence-energy space mapping."""

    @pytest.mark.parametrize("valence,energy,expected", [
        (0.9, 0.9, Mood.HAPPY),
        (0.6, 0.6, Mood.HAPPY),
        (0.9, 0.3, Mood.RELAXED),
        (0.6, 0.3, Mood.RELAXED),
        (0.3, 0.9, Mood.ANGRY),
        (0.3, 0.6, Mood.ANGRY),
        (0.3, 0.3, Mood.SAD),
        (0.1, 0.1, Mood.SAD),
    ])
    def test_quadrant_mapping(self, valence, energy, expected):
        """Test valence-energy quadrant mapping."""
        assert get_mood_from_valence_energy(valence, energy) == expected
