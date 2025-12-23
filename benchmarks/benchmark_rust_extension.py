"""Benchmark Rust extension vs pure Python implementations.

Compares performance of string similarity algorithms between Rust and Python.
"""

import sys
import time
from statistics import mean
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test data
MUSIC_ARTISTS = [
    "The Beatles",
    "Led Zeppelin",
    "Pink Floyd",
    "Queen",
    "The Rolling Stones",
    "The Who",
    "Deep Purple",
    "Black Sabbath",
    "The Doors",
    "Nirvana",
    "Radiohead",
    "Red Hot Chili Peppers",
    "Foo Fighters",
    "Green Day",
    "The Police",
    "AC/DC",
    "Eagles",
    "Fleetwood Mac",
    "Creedence Clearwater Revival",
    "The Jimi Hendrix Experience",
]

MUSIC_TITLES = [
    "Stairway to Heaven",
    "Bohemian Rhapsody",
    "Hotel California",
    "Sweet Child O' Mine",
    "Smells Like Teen Spirit",
    "Imagine",
    "Hey Jude",
    "Like a Rolling Stone",
    "Billie Jean",
    "God Save the Queen",
]

# Typos and variations for testing
VARIATIONS = [
    ("The Beatles", "Beatles, The"),
    ("The Beatles", "The Beetles"),
    ("Led Zeppelin", "Led Zepplin"),
    ("Pink Floyd", "Pink Floydd"),
    ("Radiohead", "Radio Head"),
    ("Red Hot Chili Peppers", "Red Hot Chilli Peppers"),
    ("Rolling Stones", "The Rolling Stones"),
    ("AC/DC", "AC DC"),
    ("Green Day", "Greenday"),
    ("Foo Fighters", "Foo Fghters"),
]


def benchmark_function(func, name: str, args, iterations: int = 1000):
    """Benchmark a function and return timing stats."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return {
        "name": name,
        "min": min(times),
        "max": max(times),
        "mean": mean(times),
        "total": sum(times),
        "result_sample": result,
    }


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("Rust Extension vs Pure Python Performance Comparison")
    print("=" * 70)
    print()

    # Import both implementations
    from music_organizer.utils.string_similarity import (
        StringSimilarity,
        _rust_available,
        _python_levenshtein_distance,
        _python_levenshtein_similarity,
        _python_jaro_winkler_similarity,
        _python_jaccard_similarity,
        _python_music_metadata_similarity,
    )

    print(f"Rust extension available: {_rust_available}")
    print()

    if not _rust_available:
        print("Note: Rust extension not available, comparing Python vs Python")
        print()

    # Test 1: Levenshtein Distance (single comparison)
    print("-" * 70)
    print("Test 1: Levenshtein Distance (single comparison, 1000 iterations)")
    print("-" * 70)

    s1, s2 = "Led Zeppelin", "Led Zepplin"
    py_lev = benchmark_function(
        _python_levenshtein_distance, "Python Levenshtein", (s1, s2), 1000
    )

    if _rust_available:
        rust_lev = benchmark_function(
            StringSimilarity.levenshtein_distance, "Rust Levenshtein", (s1, s2), 1000
        )
        speedup = py_lev["mean"] / rust_lev["mean"]
        print(f"Python: {py_lev['mean']*1000000:.2f} μs per call")
        print(f"Rust:   {rust_lev['mean']*1000000:.2f} μs per call")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print(f"Python: {py_lev['mean']*1000000:.2f} μs per call")
    print()

    # Test 2: Levenshtein Similarity
    print("-" * 70)
    print("Test 2: Levenshtein Similarity (single comparison, 1000 iterations)")
    print("-" * 70)

    py_sim = benchmark_function(
        _python_levenshtein_similarity, "Python Similarity", (s1, s2), 1000
    )

    if _rust_available:
        rust_sim = benchmark_function(
            StringSimilarity.levenshtein_similarity, "Rust Similarity", (s1, s2), 1000
        )
        speedup = py_sim["mean"] / rust_sim["mean"]
        print(f"Python: {py_sim['mean']*1000000:.2f} μs per call")
        print(f"Rust:   {rust_sim['mean']*1000000:.2f} μs per call")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print(f"Python: {py_sim['mean']*1000000:.2f} μs per call")
    print()

    # Test 3: Jaro-Winkler Similarity
    print("-" * 70)
    print("Test 3: Jaro-Winkler Similarity (single comparison, 1000 iterations)")
    print("-" * 70)

    py_jw = benchmark_function(
        _python_jaro_winkler_similarity, "Python Jaro-Winkler", (s1, s2), 1000
    )

    if _rust_available:
        rust_jw = benchmark_function(
            StringSimilarity.jaro_winkler_similarity, "Rust Jaro-Winkler", (s1, s2), 1000
        )
        speedup = py_jw["mean"] / rust_jw["mean"]
        print(f"Python: {py_jw['mean']*1000000:.2f} μs per call")
        print(f"Rust:   {rust_jw['mean']*1000000:.2f} μs per call")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print(f"Python: {py_jw['mean']*1000000:.2f} μs per call")
    print()

    # Test 4: Music Metadata Similarity (complex operation)
    print("-" * 70)
    print("Test 4: Music Metadata Similarity (complex, 1000 iterations)")
    print("-" * 70)

    s1, s2 = "The Beatles", "Beatles, The"
    py_meta = benchmark_function(
        _python_music_metadata_similarity, "Python Metadata", (s1, s2), 1000
    )

    if _rust_available:
        rust_meta = benchmark_function(
            StringSimilarity.music_metadata_similarity, "Rust Metadata", (s1, s2), 1000
        )
        speedup = py_meta["mean"] / rust_meta["mean"]
        print(f"Python: {py_meta['mean']*1000000:.2f} μs per call")
        print(f"Rust:   {rust_meta['mean']*1000000:.2f} μs per call")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print(f"Python: {py_meta['mean']*1000000:.2f} μs per call")
    print()

    # Test 5: Batch comparison (find best match)
    print("-" * 70)
    print("Test 5: Find Best Match (20 candidates, 100 iterations)")
    print("-" * 70)

    def py_best_match():
        best_idx = 0
        best_score = 0.0
        target = "Led Zeppelin"
        for i, artist in enumerate(MUSIC_ARTISTS):
            score = _python_levenshtein_similarity(target, artist)
            if score > best_score:
                best_score = score
                best_idx = i
        return (best_idx, best_score)

    def rust_best_match():
        return StringSimilarity.find_best_match("Led Zeppelin", MUSIC_ARTISTS)

    py_batch = benchmark_function(py_best_match, "Python Batch", (), 100)

    if _rust_available:
        rust_batch = benchmark_function(rust_best_match, "Rust Batch", (), 100)
        speedup = py_batch["mean"] / rust_batch["mean"]
        print(f"Python: {py_batch['mean']*1000:.2f} ms per call")
        print(f"Rust:   {rust_batch['mean']*1000:.2f} ms per call")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print(f"Python: {py_batch['mean']*1000:.2f} ms per call")
    print()

    # Test 6: All-pairs comparison (O(n^2) complexity)
    print("-" * 70)
    print("Test 6: All-Pairs Similarity (20 strings, 10 iterations)")
    print("-" * 70)

    def py_all_pairs():
        results = []
        for i in range(len(MUSIC_ARTISTS)):
            for j in range(i + 1, len(MUSIC_ARTISTS)):
                sim = _python_levenshtein_similarity(
                    MUSIC_ARTISTS[i], MUSIC_ARTISTS[j]
                )
                if sim > 0.5:
                    results.append((i, j, sim))
        return results

    def rust_all_pairs():
        return StringSimilarity.find_similar_pairs(MUSIC_ARTISTS, 0.5)

    py_pairs = benchmark_function(py_all_pairs, "Python All-Pairs", (), 10)

    if _rust_available:
        rust_pairs = benchmark_function(rust_all_pairs, "Rust All-Pairs", (), 10)
        speedup = py_pairs["mean"] / rust_pairs["mean"]
        print(f"Python: {py_pairs['mean']:.4f} s per call")
        print(f"Rust:   {rust_pairs['mean']:.4f} s per call")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Pairs found: Python={len(py_pairs['result_sample'])}, "
              f"Rust={len(rust_pairs['result_sample'])}")
    else:
        print(f"Python: {py_pairs['mean']:.4f} s per call")
        print(f"Pairs found: {len(py_pairs['result_sample'])}")
    print()

    # Summary table
    print("=" * 70)
    print("Summary of Results")
    print("=" * 70)
    print()
    print(f"{'Algorithm':<30} {'Python (μs)':<15} {'Rust (μs)':<15} {'Speedup':<10}")
    print("-" * 70)

    if _rust_available:
        results = [
            ("Levenshtein Distance", py_lev["mean"] * 1e6, rust_lev["mean"] * 1e6),
            ("Levenshtein Similarity", py_sim["mean"] * 1e6, rust_sim["mean"] * 1e6),
            ("Jaro-Winkler", py_jw["mean"] * 1e6, rust_jw["mean"] * 1e6),
            ("Music Metadata", py_meta["mean"] * 1e6, rust_meta["mean"] * 1e6),
            ("Find Best Match", py_batch["mean"] * 1e3, rust_batch["mean"] * 1e3),
            ("All Pairs (ms)", py_pairs["mean"] * 1e3, rust_pairs["mean"] * 1e3),
        ]

        for name, py_time, rust_time in results:
            speedup = py_time / rust_time if rust_time > 0 else 0
            unit = "μs" if "ms" not in name else "ms"
            print(f"{name:<30} {py_time:<15.2f} {rust_time:<15.2f} {speedup:<10.2f}x")
    else:
        print("Rust extension not available. Run with: maturin develop")
        print()
        print("To build and install the Rust extension:")
        print("  cd native/music_organizer_rs")
        print("  pip install maturin")
        print("  maturin develop --release")

    print()
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
