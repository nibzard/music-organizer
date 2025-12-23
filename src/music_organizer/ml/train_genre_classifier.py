"""Training script for genre classifier.

This module provides utilities to train the genre classifier using
MusicBrainz or other public datasets.
"""

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .genre_classifier import GenreClassifier, TOP_GENRES, _extract_features


class GenreClassifierTrainer:
    """Trainer for the genre classification model."""

    def __init__(
        self,
        output_path: Optional[Path] = None,
        max_features: int = 5000,
        test_size: float = 0.2,
    ):
        """Initialize the trainer.

        Args:
            output_path: Where to save the trained model
            max_features: Max features for TF-IDF vectorizer
            test_size: Fraction of data to use for testing
        """
        self.output_path = output_path or GenreClassifier().model_path
        self.max_features = max_features
        self.test_size = test_size

        # Model components
        self._vectorizer: Any = None
        self._model: Any = None
        self._genre_labels: List[str] = TOP_GENRES

    def train_from_csv(
        self,
        csv_path: Path,
        text_col: str = "text",
        label_col: str = "genre",
    ) -> Dict[str, float]:
        """Train model from CSV file.

        Expected CSV format:
            text,genre
            "Led Zeppelin Stairway to Heaven",rock
            "Daft Punk Around the World",electronic

        Args:
            csv_path: Path to CSV file
            text_col: Name of column containing text features
            label_col: Name of column containing genre labels

        Returns:
            Dictionary with training metrics
        """
        texts = []
        labels = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row[text_col])
                labels.append(row[label_col])

        return self._train(texts, labels)

    def train_from_json(self, json_path: Path) -> Dict[str, float]:
        """Train model from JSON file.

        Expected JSON format:
        [
            {"text": "...", "genre": "rock"},
            {"text": "...", "genre": "electronic"}
        ]

        Args:
            json_path: Path to JSON file

        Returns:
            Dictionary with training metrics
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [item["text"] for item in data]
        labels = [item["genre"] for item in data]

        return self._train(texts, labels)

    def train_from_musicbrainz_dump(
        self,
        dump_path: Path,
    ) -> Dict[str, float]:
        """Train model from MusicBrainz JSON dump.

        Args:
            dump_path: Path to MusicBrainz dump file

        Returns:
            Dictionary with training metrics
        """
        texts = []
        labels = []

        with open(dump_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Extract text features
                    text = " ".join(filter(None, [
                        record.get("title", ""),
                        record.get("artist", ""),
                        record.get("album", ""),
                    ]))

                    # Get genre tags
                    genres = record.get("genres", [])
                    if genres and text:
                        texts.append(text)
                        # Use first genre as primary
                        labels.append(genres[0])
                except (json.JSONDecodeError, KeyError):
                    continue

        return self._train(texts, labels)

    def train_from_musicbrainz_api(
        self,
        artist_limit: int = 1000,
        queries: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Train model by fetching data from MusicBrainz API.

        Args:
            artist_limit: Maximum number of artists to fetch
            queries: List of genre queries (defaults to TOP_GENRES)

        Returns:
            Dictionary with training metrics
        """
        queries = queries or TOP_GENRES

        try:
            import musicbrainzngs as mb
        except ImportError:
            raise ImportError(
                "Install musicbrainzngs: pip install musicbrainzngs"
            )

        # Set user agent (required by MusicBrainz)
        mb.set_useragent("music-organizer", "1.0", "https://github.com/nibzard/music-organizer")

        texts = []
        labels = []

        for genre in queries:
            # Search for artists in this genre
            result = mb.search_artists(
                query=f"genre:{genre}",
                limit=min(100, artist_limit // len(queries)),
            )

            for artist in result.get("artist-list", []):
                text_parts = [
                    artist.get("name", ""),
                ]
                # Add some release names
                if "release-list" in artist:
                    for release in artist["release-list"][:3]:
                        text_parts.append(release.get("title", ""))

                text = " ".join(filter(None, text_parts))
                if text:
                    texts.append(text)
                    labels.append(genre)

        return self._train(texts, labels)

    def _train(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """Internal training method.

        Args:
            texts: List of text samples
            labels: List of genre labels

        Returns:
            Dictionary with training metrics
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import classification_report, accuracy_score
        except ImportError:
            raise ImportError(
                "Install scikit-learn: pip install scikit-learn"
            )

        # Filter to only top genres
        valid_indices = [
            i for i, label in enumerate(labels)
            if label in self._genre_labels
        ]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]

        if not texts:
            raise ValueError("No valid training data found")

        # Create vectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words="english",
        )

        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(self._genre_labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=self.test_size,
            random_state=42,
            stratify=labels,
        )

        # Vectorize
        X_train_vec = self._vectorizer.fit_transform(X_train)
        X_test_vec = self._vectorizer.transform(X_test)

        # Train model
        self._model = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            class_weight="balanced",
        )
        self._model.fit(X_train_vec, label_encoder.transform(y_train))

        # Evaluate
        y_pred = self._model.predict(X_test_vec)
        accuracy = accuracy_score(label_encoder.transform(y_test), y_pred)

        # Get per-class metrics
        report = classification_report(
            label_encoder.transform(y_test),
            y_pred,
            target_names=self._genre_labels,
            output_dict=True,
            zero_division=0,
        )

        # Save model
        self._save()

        return {
            "accuracy": accuracy,
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "classes": len(self._genre_labels),
            "report": report,
        }

    def _save(self) -> None:
        """Save the trained model."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "vectorizer": self._vectorizer,
                "genres": self._genre_labels,
                "metadata": {
                    "max_features": self.max_features,
                    "model_type": "logistic_regression",
                },
            }, f)


def create_synthetic_training_data(output_path: Path, samples_per_genre: int = 50) -> None:
    """Create synthetic training data for testing.

    This creates a basic training dataset using genre-specific patterns.
    For production use, you should use real data from MusicBrainz or similar.

    Args:
        output_path: Path to save the training data CSV
        samples_per_genre: Number of samples to generate per genre
    """
    import random

    # Sample data templates
    templates = {
        "electronic": [
            ("{artist} - {title} (Original Mix)", "electronic"),
            ("{artist} {title} Remix", "electronic"),
            ("DJ {artist} - Club Mix {title}", "electronic"),
        ],
        "rock": [
            ("{artist} - {title}", "rock"),
            ("{artist} Band - {title} (Live)", "rock"),
            ("{artist} - {title} [Rock Version]", "rock"),
        ],
        "hiphop": [
            ("{artist} - {title} (feat. {artist2})", "hiphop"),
            ("{artist} - {title}", "hiphop"),
        ],
        "jazz": [
            ("{artist} Quartet - {title}", "jazz"),
            ("{artist} - {title} (Jazz Standard)", "jazz"),
        ],
        "classical": [
            ("{composer} - {piece}", "classical"),
            ("{artist} Orchestra - {piece}", "classical"),
        ],
    }

    artists = {
        "electronic": ["Daft Punk", "Deadmau5", "Skrillex", "Calvin Harris"],
        "rock": ["Led Zeppelin", "Pink Floyd", "The Beatles", "Queen"],
        "hiphop": ["Kendrick Lamar", "Drake", "J Cole", "Kanye West"],
        "jazz": ["Miles Davis", "John Coltrane", "Bill Evans", "Charlie Parker"],
        "classical": ["Beethoven", "Mozart", "Bach", "Chopin"],
    }

    titles = {
        "electronic": ["Around the World", "Strobe", "Bangarang", "Summer"],
        "rock": ["Stairway to Heaven", "Comfortably Numb", "Bohemian Rhapsody", "Hotel California"],
        "hiphop": ["HUMBLE", "God's Plan", "No Role Modelz", "Stronger"],
        "jazz": ["So What", "Giant Steps", "Blue in Green", "Ornithology"],
        "classical": ["Symphony No. 5", "Moonlight Sonata", "Cello Suite", "Nocturne Op.9"],
    }

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "genre"])

        for genre, genre_templates in templates.items():
            for _ in range(samples_per_genre):
                template = random.choice(genre_templates)
                artist = random.choice(artists[genre])
                title = random.choice(titles[genre])

                text = template.format(
                    artist=artist,
                    title=title,
                    artist2=random.choice(artists[genre]),
                    composer=random.choice(artists[genre]),
                    piece=random.choice(titles[genre]),
                )

                writer.writerow([text, genre])

    print(f"Created synthetic training data: {output_path}")
    print(f"Samples: {len(templates) * samples_per_genre}")
