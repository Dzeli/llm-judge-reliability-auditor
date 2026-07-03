from __future__ import annotations

import difflib
import re
from collections import Counter
from models.validation import MutationValidation

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "as", "by", "that", "this", "it", "from", "at", "which", "when", "what",
}


def validate_verbosity(original: str, mutated: str, variant_id: str, min_ratio: float = 1.8) -> MutationValidation:
    ow = _words(original)
    mw = _words(mutated)
    ratio = len(mw) / max(len(ow), 1)
    similarity = difflib.SequenceMatcher(None, original.lower(), mutated.lower()).ratio()
    new_terms = _new_content_terms(ow, mw)

    warnings: list[str] = []
    if ratio < min_ratio:
        warnings.append(f"Length ratio {ratio:.2f} is below target {min_ratio:.2f}.")
    if len(new_terms) > max(6, len(set(ow)) * 0.45):
        warnings.append("Mutation may have introduced many new content terms.")

    return MutationValidation(
        variant_id=variant_id,
        is_valid=not warnings,
        metrics={
            "original_words": len(ow),
            "mutated_words": len(mw),
            "length_ratio": round(ratio, 3),
            "sequence_similarity": round(similarity, 3),
            "new_content_terms": len(new_terms),
        },
        warnings=warnings,
    )


def validate_style(original: str, mutated: str, variant_id: str, expected_style: str) -> MutationValidation:
    ow = _words(original)
    mw = _words(mutated)
    similarity = difflib.SequenceMatcher(None, original.lower(), mutated.lower()).ratio()
    new_terms = _new_content_terms(ow, mw)
    avg_sentence_len = _avg_sentence_len(mutated)
    length_ratio = len(mw) / max(len(ow), 1)

    warnings: list[str] = []
    if len(new_terms) > max(6, len(set(ow)) * 0.5):
        warnings.append("Style rewrite may have introduced too many new content terms.")
    # Important V3 control: style should not secretly become a verbosity test.
    if not 0.75 <= length_ratio <= 1.35:
        warnings.append(f"Style rewrite length ratio {length_ratio:.2f} may confound style with verbosity.")
    if expected_style == "plain" and avg_sentence_len > 28:
        warnings.append("Plain rewrite still has long sentences.")
    if expected_style == "polished" and len(mw) < max(8, len(ow) * 0.75):
        warnings.append("Polished rewrite is unexpectedly shorter than original.")

    return MutationValidation(
        variant_id=variant_id,
        is_valid=not warnings,
        metrics={
            "original_words": len(ow),
            "mutated_words": len(mw),
            "length_ratio": round(length_ratio, 3),
            "sequence_similarity": round(similarity, 3),
            "new_content_terms": len(new_terms),
            "avg_sentence_len": round(avg_sentence_len, 2),
            "expected_style": expected_style,
        },
        warnings=warnings,
    )


def validate_rubric_paraphrase(original: str, paraphrase: str, variant_id: str) -> MutationValidation:
    ow = _content_counter(original)
    pw = _content_counter(paraphrase)
    overlap = len(set(ow) & set(pw)) / max(len(set(ow) | set(pw)), 1)
    warnings: list[str] = []
    if not paraphrase.strip():
        warnings.append("Empty paraphrase generated.")
    if overlap < 0.12:
        warnings.append("Very low lexical overlap; paraphrase may have shifted meaning.")
    return MutationValidation(
        variant_id=variant_id,
        is_valid=not warnings,
        metrics={"content_term_jaccard": round(overlap, 3)},
        warnings=warnings,
    )


def validate_rubric_priority_shift(original: str, shifted: str, variant_id: str) -> MutationValidation:
    """Validate a priority-shift rubric variant.

    Unlike a paraphrase, a priority-shift SHOULD differ from the original — it
    deliberately reorders criterion importance. We warn when the variant is empty
    (generation failed), identical to the original (no shift occurred), or has such
    low overlap that it has dropped criteria entirely rather than reweighting them.
    """
    ow = _content_counter(original)
    sw = _content_counter(shifted)
    overlap = len(set(ow) & set(sw)) / max(len(set(ow) | set(sw)), 1)
    warnings: list[str] = []
    if not shifted.strip():
        warnings.append("Empty priority-shift rubric generated.")
    elif original.strip().lower() == shifted.strip().lower():
        warnings.append("Priority-shift rubric is identical to original; no shift occurred.")
    if overlap < 0.10:
        warnings.append("Very low lexical overlap; priority shift may have removed criteria rather than reweighting them.")
    return MutationValidation(
        variant_id=variant_id,
        is_valid=not warnings,
        metrics={"content_term_jaccard": round(overlap, 3)},
        warnings=warnings,
    )


def not_applicable(variant_id: str) -> MutationValidation:
    return MutationValidation(variant_id=variant_id, is_valid=True, validation_type="not_applicable")


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_'-]+", text.lower())


def _content_counter(text_or_words) -> Counter:
    words = text_or_words if isinstance(text_or_words, list) else _words(text_or_words)
    return Counter(w for w in words if w not in STOPWORDS and len(w) > 2)


def _new_content_terms(original_words: list[str], mutated_words: list[str]) -> set[str]:
    original = set(_content_counter(original_words))
    mutated = set(_content_counter(mutated_words))
    return mutated - original


def _avg_sentence_len(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return float(len(_words(text)))
    return sum(len(_words(s)) for s in sentences) / len(sentences)
