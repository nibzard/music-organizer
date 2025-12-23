"""Training script for mood classifier.

This module provides utilities to train the mood classifier using
audio features extracted from music files.
"""

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .mood_classifier import MoodClassifier, MOOD_DEFINITIONS
from music_organizer.domain.classification.value_objects import Mood


class MoodClassifierTrainer:
    """Trainer for the mood classification model.

    The mood classifier can be trained on datasets that include:
    - Audio features (valence, energy, danceability, acousticness)
    - Mood labels (happy, sad, energetic, calm, etc.)
    """

    def __init__(
        self,
        output_path: Optional[Path] = None,
        test_size: float = 0.2,
    ):
        """Initialize the trainer.

        Args:
            output_path: Where to save the trained model
            test_size: Fraction of data to use for testing
        """
        self.output_path = output_path or MoodClassifier().model_path
        self.test_size = test_size

        # Model components
        self._model: Any = None
        self._scaler: Any = None
        self._mood_labels: List[str] = [m.value for m in Mood]

    def train_from_csv(
        self,
        csv_path: Path,
        feature_cols: List[str] = ["valence", "energy", "danceability", "acousticness"],
        label_col: str = "mood",
    ) -> Dict[str, float]:
        """Train model from CSV file.

        Expected CSV format:
            valence,energy,danceability,acousticness,mood
            0.8,0.7,0.6,0.3,happy
            0.2,0.3,0.2,0.8,sad

        Args:
            csv_path: Path to CSV file
            feature_cols: Names of columns containing audio features
            label_col: Name of column containing mood labels

        Returns:
            Dictionary with training metrics
        """
        features = []
        labels = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature_vector = [float(row.get(col, 0)) for col in feature_cols]
                features.append(feature_vector)
                labels.append(row[label_col])

        return self._train(features, labels, feature_cols)

    def train_from_json(self, json_path: Path) -> Dict[str, float]:
        """Train model from JSON file.

        Expected JSON format:
        [
            {"valence": 0.8, "energy": 0.7, "danceability": 0.6, "acousticness": 0.3, "mood": "happy"},
            {"valence": 0.2, "energy": 0.3, "danceability": 0.2, "acousticness": 0.8, "mood": "sad"}
        ]

        Args:
            json_path: Path to JSON file

        Returns:
            Dictionary with training metrics
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        feature_cols = ["valence", "energy", "danceability", "acousticness"]
        features = []
        labels = []

        for item in data:
            feature_vector = [float(item.get(col, 0)) for col in feature_cols]
            features.append(feature_vector)
            labels.append(item["mood"])

        return self._train(features, labels, feature_cols)

    def train_from_audio_features(
        self,
        audio_features_list: List[Tuple[Dict[str, float], Mood]],
    ) -> Dict[str, float]:
        """Train model from list of audio features with mood labels.

        Args:
            audio_features_list: List of (features_dict, mood) tuples
                features_dict should contain: valence, energy, danceability, acousticness

        Returns:
            Dictionary with training metrics
        """
        feature_cols = ["valence", "energy", "danceability", "acousticness"]
        features = []
        labels = []

        for features_dict, mood in audio_features_list:
            feature_vector = [
                float(features_dict.get("valence", 0.5)),
                float(features_dict.get("energy", 0.5)),
                float(features_dict.get("danceability", 0.5)),
                float(features_dict.get("acousticness", 0.5)),
            ]
            features.append(feature_vector)
            labels.append(mood.value)

        return self._train(features, labels, feature_cols)

    def _train(
        self,
        features: List[List[float]],
        labels: List[str],
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Internal training method.

        Args:
            features: List of feature vectors
            labels: List of mood labels
            feature_names: Names of features

        Returns:
            Dictionary with training metrics
        """
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import classification_report, accuracy_score
        except ImportError:
            raise ImportError(
                "Install scikit-learn: pip install scikit-learn"
            )

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        # Filter to only known moods
        valid_indices = [
            i for i, label in enumerate(y)
            if label in self._mood_labels
        ]
        X = X[valid_indices]
        y = y[valid_indices]

        if len(X) == 0:
            raise ValueError("No valid training data found")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=42,
            stratify=y,
        )

        # Scale features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        # Train model - Random Forest works well for mood classification
        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
        )
        self._model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self._model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Get per-class metrics
        report = classification_report(
            y_test,
            y_pred,
            target_names=self._mood_labels,
            output_dict=True,
            zero_division=0,
        )

        # Get feature importances
        importances = dict(zip(feature_names, self._model.feature_importances_))

        # Save model
        self._save(feature_names)

        return {
            "accuracy": accuracy,
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "classes": len(self._mood_labels),
            "feature_importances": importances,
            "report": report,
        }

    def _save(self, feature_names: List[str]) -> None:
        """Save the trained model."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "scaler": self._scaler,
                "moods": self._mood_labels,
                "feature_names": feature_names,
                "metadata": {
                    "model_type": "random_forest",
                    "test_size": self.test_size,
                },
            }, f)


def create_synthetic_mood_data(output_path: Path, samples_per_mood: int = 50) -> None:
    """Create synthetic training data for mood classification.

    This creates a basic training dataset using mood-specific feature patterns.
    For production use, you should use real labeled data.

    Args:
        output_path: Path to save the training data CSV
        samples_per_mood: Number of samples to generate per mood
    """
    import random

    # Feature ranges for each mood
    mood_ranges = {
        "happy": {
            "valence": (0.6, 1.0),
            "energy": (0.5, 1.0),
            "danceability": (0.5, 1.0),
            "acousticness": (0.0, 0.7),
        },
        "sad": {
            "valence": (0.0, 0.4),
            "energy": (0.0, 0.5),
            "danceability": (0.0, 0.5),
            "acousticness": (0.2, 1.0),
        },
        "energetic": {
            "valence": (0.4, 1.0),
            "energy": (0.7, 1.0),
            "danceability": (0.6, 1.0),
            "acousticness": (0.0, 0.5),
        },
        "calm": {
            "valence": (0.4, 0.8),
            "energy": (0.0, 0.4),
            "danceability": (0.0, 0.4),
            "acousticness": (0.3, 1.0),
        },
        "angry": {
            "valence": (0.0, 0.4),
            "energy": (0.6, 1.0),
            "danceability": (0.3, 0.8),
            "acousticness": (0.0, 0.4),
        },
        "relaxed": {
            "valence": (0.5, 1.0),
            "energy": (0.0, 0.4),
            "danceability": (0.0, 0.4),
            "acousticness": (0.3, 1.0),
        },
        "melancholic": {
            "valence": (0.2, 0.5),
            "energy": (0.0, 0.5),
            "danceability": (0.0, 0.4),
            "acousticness": (0.2, 1.0),
        },
        "uplifting": {
            "valence": (0.7, 1.0),
            "energy": (0.5, 1.0),
            "danceability": (0.4, 1.0),
            "acousticness": (0.0, 0.6),
        },
        "dark": {
            "valence": (0.0, 0.4),
            "energy": (0.3, 0.8),
            "danceability": (0.2, 0.6),
            "acousticness": (0.0, 0.6),
        },
        "bright": {
            "valence": (0.6, 1.0),
            "energy": (0.3, 0.8),
            "danceability": (0.2, 0.7),
            "acousticness": (0.0, 0.8),
        },
    }

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["valence", "energy", "danceability", "acousticness", "mood"])

        for mood, ranges in mood_ranges.items():
            for _ in range(samples_per_mood):
                # Generate features within the mood's range with some noise
                valence = random.uniform(*ranges["valence"])
                energy = random.uniform(*ranges["energy"])
                danceability = random.uniform(*ranges["danceability"])
                acousticness = random.uniform(*ranges["acousticness"])

                writer.writerow([valence, energy, danceability, acousticness, mood])

    print(f"Created synthetic mood training data: {output_path}")
    print(f"Samples: {len(mood_ranges) * samples_per_mood}")


def train_model_from_spotify_dataset(
    spotify_data_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Train mood classifier from Spotify dataset.

    The Spotify dataset contains audio features that map well to mood.
    This function expects a CSV with columns:
    - valence, energy, danceability, acousticness
    - And a mood column or genre column that can be mapped to mood

    Args:
        spotify_data_path: Path to Spotify dataset CSV
        output_path: Where to save the trained model

    Returns:
        Dictionary with training metrics
    """
    # Map Spotify genres/tracks to moods
    genre_to_mood = {
        "pop": "happy",
        "rock": "energetic",
        "metal": "angry",
        "classical": "calm",
        "jazz": "relaxed",
        "blues": "melancholic",
        "electronic": "energetic",
        "hip-hop": "energetic",
        "country": "happy",
        "ambient": "calm",
    }

    trainer = MoodClassifierTrainer(output_path=output_path)

    # Load and process data
    import pandas as pd
    df = pd.read_csv(spotify_data_path)

    # Map genre to mood if mood column not present
    if "mood" not in df.columns and "genre" in df.columns:
        df["mood"] = df["genre"].map(genre_to_mood)

    # Filter to required columns
    required_cols = ["valence", "energy", "danceability", "acousticness", "mood"]
    df = df[required_cols].dropna()

    # Train
    return trainer.train_from_csv(
        spotify_data_path,
        feature_cols=["valence", "energy", "danceability", "acousticness"],
        label_col="mood",
    )
