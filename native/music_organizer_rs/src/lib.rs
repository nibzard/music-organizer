//! Music Organizer Rust Extensions
//!
//! High-performance string similarity and distance calculations
//! optimized for music metadata matching and duplicate detection.

use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashSet;

/// Levenshtein distance implementation using optimized algorithm
///
/// Returns the minimum number of single-character edits (insertions, deletions, or substitutions)
/// required to change one string into another.
#[pyfunction]
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    if s1.is_empty() {
        return s2.chars().count();
    }
    if s2.is_empty() {
        return s1.chars().count();
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    // Use smaller array for optimization
    if len1 < len2 {
        return levenshtein_distance(s2, s1);
    }

    // Single row array for space optimization
    let mut previous: Vec<usize> = (0..=len2).collect();
    let mut current: Vec<usize> = vec![0; len2 + 1];

    for (i, c1) in s1_chars.iter().enumerate() {
        current[0] = i + 1;

        for (j, c2) in s2_chars.iter().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            current[j + 1] = [
                current[j] + 1,           // insertion
                previous[j + 1] + 1,      // deletion
                previous[j] + cost,       // substitution
            ]
            .into_iter()
            .min()
            .unwrap();
        }

        std::mem::swap(&mut previous, &mut current);
    }

    previous[len2]
}

/// Normalized Levenshtein similarity (0.0 to 1.0)
///
/// Returns a similarity score where 1.0 means identical strings
/// and 0.0 means completely different.
#[pyfunction]
fn levenshtein_similarity(s1: &str, s2: &str) -> f64 {
    if s1 == s2 {
        return 1.0;
    }

    let max_len = s1.chars().count().max(s2.chars().count());
    if max_len == 0 {
        return 1.0;
    }

    let distance = levenshtein_distance(s1, s2);
    1.0 - (distance as f64 / max_len as f64)
}

/// Damerau-Levenshtein distance (includes transpositions)
///
/// Counts transpositions of adjacent characters as a single edit operation.
#[pyfunction]
fn damerau_levenshtein_distance(s1: &str, s2: &str) -> usize {
    if s1.is_empty() {
        return s2.chars().count();
    }
    if s2.is_empty() {
        return s1.chars().count();
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    let inf = len1 + len2;
    let mut d: Vec<Vec<usize>> = vec![vec![inf; len2 + 2]; len1 + 2];

    d[0][0] = inf;

    for i in 0..=len1 {
        d[i + 1][1] = i;
        d[i + 1][0] = inf;
    }

    for j in 0..=len2 {
        d[1][j + 1] = j;
        d[0][j + 1] = inf;
    }

    let mut da: std::collections::HashMap<char, usize> = std::collections::HashMap::new();

    for i in 1..=len1 {
        let mut db = 0;
        for j in 1..=len2 {
            let i2 = da.get(&s2_chars[j - 1]).copied().unwrap_or(0);
            let j2 = db;

            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            d[i + 1][j + 1] = [
                d[i][j] + cost,           // substitution
                d[i + 1][j] + 1,          // insertion
                d[i][j + 1] + 1,          // deletion
                d[i2][j2] + (i - i2 - 1) + 1 + (j - j2 - 1), // transposition
            ]
            .into_iter()
            .min()
            .unwrap();

            if s1_chars[i - 1] == s2_chars[j - 1] {
                db = j;
            }
        }

        da.insert(s1_chars[i - 1], i);
    }

    d[len1 + 1][len2 + 1]
}

/// Jaro-Winkler similarity
///
/// Gives higher scores to strings that match from the beginning.
/// Useful for detecting typos and name variations.
#[pyfunction]
fn jaro_winkler_similarity(s1: &str, s2: &str) -> f64 {
    let jaro = jaro_similarity(s1, s2);

    // Calculate prefix length (max 4 characters)
    let prefix_len = s1
        .chars()
        .zip(s2.chars())
        .take_while(|(c1, c2)| c1 == c2)
        .take(4)
        .count();

    let prefix_scale = 0.1;
    let jaro_winkler = jaro + (prefix_len as f64 * prefix_scale * (1.0 - jaro));

    jaro_winkler.min(1.0)
}

/// Jaro similarity
#[pyfunction]
fn jaro_similarity(s1: &str, s2: &str) -> f64 {
    if s1 == s2 {
        return 1.0;
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();

    if len1 == 0 || len2 == 0 {
        return 0.0;
    }

    // Match distance
    let match_distance = len1.max(len2) / 2 - 1;
    let max_match_distance = match_distance.max(0);

    let mut s1_matches = vec![false; len1];
    let mut s2_matches = vec![false; len2];
    let mut matches = 0;

    for i in 0..len1 {
        let start = i.saturating_sub(max_match_distance);
        let end = (i + max_match_distance + 1).min(len2);

        for j in start..end {
            if !s2_matches[j] && s1_chars[i] == s2_chars[j] {
                s1_matches[i] = true;
                s2_matches[j] = true;
                matches += 1;
                break;
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut transpositions = 0;
    let mut k = 0;

    for i in 0..len1 {
        if s1_matches[i] {
            while !s2_matches[k] {
                k += 1;
            }
            if s1_chars[i] != s2_chars[k] {
                transpositions += 1;
            }
            k += 1;
        }
    }

    let transpositions = transpositions / 2;

    // Calculate Jaro similarity
    let m = matches as f64;
    (m / (len1 as f64) + m / (len2 as f64) + (m - transpositions as f64) / m) / 3.0
}

/// Jaccard similarity for word sets
///
/// Measures similarity between sets of words (tokens).
/// Useful for comparing titles, phrases, etc.
#[pyfunction]
fn jaccard_similarity(s1: &str, s2: &str) -> f64 {
    let words1: HashSet<&str> = s1.split_whitespace().collect();
    let words2: HashSet<&str> = s2.split_whitespace().collect();

    if words1.is_empty() && words2.is_empty() {
        return 1.0;
    }

    if words1.is_empty() || words2.is_empty() {
        return 0.0;
    }

    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f64 / union as f64
}

/// Cosine similarity for word frequency vectors
#[pyfunction]
fn cosine_similarity(s1: &str, s2: &str) -> f64 {
    use std::collections::HashMap;

    if s1 == s2 {
        return 1.0;
    }

    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }

    let words1: Vec<&str> = s1.split_whitespace().collect();
    let words2: Vec<&str> = s2.split_whitespace().collect();

    if words1.is_empty() || words2.is_empty() {
        return 0.0;
    }

    // Build frequency maps
    let mut freq1: HashMap<&str, usize> = HashMap::new();
    let mut freq2: HashMap<&str, usize> = HashMap::new();

    for word in &words1 {
        *freq1.entry(word).or_insert(0) += 1;
    }
    for word in &words2 {
        *freq2.entry(word).or_insert(0) += 1;
    }

    // Calculate dot product and magnitudes
    let mut dot_product: f64 = 0.0;
    let mut mag1: f64 = 0.0;
    let mut mag2: f64 = 0.0;

    for (word, f1) in &freq1 {
        let f2 = freq2.get(word).unwrap_or(&0);
        dot_product += (*f1 as f64) * (*f2 as f64);
        mag1 += (*f1 as f64) * (*f1 as f64);
    }

    for f2 in freq2.values() {
        mag2 += (*f2 as f64) * (*f2 as f64);
    }

    mag1 = mag1.sqrt();
    mag2 = mag2.sqrt();

    if mag1 == 0.0 || mag2 == 0.0 {
        return 0.0;
    }

    dot_product / (mag1 * mag2)
}

/// Sorensen-Dice coefficient
///
/// Similar to Jaccard but gives more weight to matches.
#[pyfunction]
fn sorensen_dice_similarity(s1: &str, s2: &str) -> f64 {
    if s1.len() < 2 || s2.len() < 2 {
        if s1 == s2 {
            return 1.0;
        }
        return 0.0;
    }

    let bigrams1: HashSet<&str> = s1.as_bytes().windows(2).map(|w| std::str::from_utf8(w).unwrap_or("")).collect();
    let bigrams2: HashSet<&str> = s2.as_bytes().windows(2).map(|w| std::str::from_utf8(w).unwrap_or("")).collect();

    if bigrams1.is_empty() && bigrams2.is_empty() {
        return 1.0;
    }

    if bigrams1.is_empty() || bigrams2.is_empty() {
        return 0.0;
    }

    let intersection = bigrams1.intersection(&bigrams2).count();

    if bigrams1.len() + bigrams2.len() == 0 {
        return 0.0;
    }

    (2.0 * intersection as f64) / ((bigrams1.len() + bigrams2.len()) as f64)
}

/// Find best match from a list of candidates
///
/// Returns the index and similarity score of the best matching string.
#[pyfunction]
fn find_best_match(target: &str, candidates: Vec<String>) -> PyResult<(usize, f64)> {
    if candidates.is_empty() {
        return Ok((0, 0.0));
    }

    let mut best_index = 0;
    let mut best_score = 0.0;

    for (i, s) in candidates.iter().enumerate() {
        let score = levenshtein_similarity(target, s);
        if score > best_score {
            best_score = score;
            best_index = i;
        }
    }

    Ok((best_index, best_score))
}

/// Batch calculate similarities between all pairs
///
/// Returns a list of (index1, index2, similarity) tuples for pairs
/// with similarity above the threshold.
#[pyfunction]
fn find_similar_pairs(strings: Vec<String>, threshold: f64) -> PyResult<Vec<(usize, usize, f64)>> {
    let mut results = Vec::new();

    for i in 0..strings.len() {
        for j in (i + 1)..strings.len() {
            let sim = levenshtein_similarity(&strings[i], &strings[j]);
            if sim >= threshold {
                results.push((i, j, sim));
            }
        }
    }

    Ok(results)
}

/// Fuzzy string match that returns all similarity metrics
#[pyfunction]
fn fuzzy_match(s1: &str, s2: &str) -> PyResult<FuzzyMatchResult> {
    Ok(FuzzyMatchResult {
        levenshtein_distance: levenshtein_distance(s1, s2) as f64,
        levenshtein_similarity: levenshtein_similarity(s1, s2),
        jaro_winkler_similarity: jaro_winkler_similarity(s1, s2),
        jaccard_similarity: jaccard_similarity(s1, s2),
        cosine_similarity: cosine_similarity(s1, s2),
        sorensen_dice_similarity: sorensen_dice_similarity(s1, s2),
    })
}

/// Result struct for fuzzy match
#[pyclass]
#[derive(Clone)]
pub struct FuzzyMatchResult {
    #[pyo3(get)]
    pub levenshtein_distance: f64,
    #[pyo3(get)]
    pub levenshtein_similarity: f64,
    #[pyo3(get)]
    pub jaro_winkler_similarity: f64,
    #[pyo3(get)]
    pub jaccard_similarity: f64,
    #[pyo3(get)]
    pub cosine_similarity: f64,
    #[pyo3(get)]
    pub sorensen_dice_similarity: f64,
}

#[pymethods]
impl FuzzyMatchResult {
    /// Get the best similarity score among all metrics
    fn best_score(&self) -> f64 {
        self.levenshtein_similarity
            .max(self.jaro_winkler_similarity)
            .max(self.jaccard_similarity)
            .max(self.cosine_similarity)
            .max(self.sorensen_dice_similarity)
    }

    fn __repr__(&self) -> String {
        format!(
            "FuzzyMatchResult(levenshtein={:.3}, jaro_winkler={:.3}, jaccard={:.3})",
            self.levenshtein_similarity, self.jaro_winkler_similarity, self.jaccard_similarity
        )
    }
}

/// Music metadata specific comparison
///
/// Optimized for comparing artist names, titles, etc.
/// Handles common variations like "The Beatles", "Beatles, The".
#[pyfunction]
fn music_metadata_similarity(s1: &str, s2: &str) -> f64 {
    // Normalize strings
    let norm1 = normalize_music_text(s1);
    let norm2 = normalize_music_text(s2);

    if norm1 == norm2 {
        return 1.0;
    }

    // Try multiple similarity measures
    let lev_sim = levenshtein_similarity(&norm1, &norm2);
    let jw_sim = jaro_winkler_similarity(&norm1, &norm2);
    let dice_sim = sorensen_dice_similarity(&norm1, &norm2);

    // Weighted average (Jaro-Winkler is most accurate for names)
    lev_sim * 0.3 + jw_sim * 0.5 + dice_sim * 0.2
}

/// Normalize music metadata text for comparison
fn normalize_music_text(s: &str) -> String {
    // Remove leading "The " and similar prefixes
    let trimmed = s.trim().to_lowercase();
    let cleaned = trimmed
        .strip_prefix("the ")
        .unwrap_or(&trimmed)
        .strip_prefix("a ")
        .unwrap_or(&trimmed)
        .strip_prefix("an ")
        .unwrap_or(&trimmed);

    // Remove special characters and extra whitespace
    let mut normalized = String::new();
    for c in cleaned.chars() {
        if c.is_alphanumeric() || c.is_whitespace() {
            normalized.push(c);
        }
    }

    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Python module definition
#[pymodule]
fn music_organizer_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_distance, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(sorensen_dice_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(find_best_match, m)?)?;
    m.add_function(wrap_pyfunction!(find_similar_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy_match, m)?)?;
    m.add_function(wrap_pyfunction!(music_metadata_similarity, m)?)?;
    m.add_class::<FuzzyMatchResult>()?;

    Ok(())
}
