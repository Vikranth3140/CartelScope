"""Prompt templates for citation-intent teacher labeling."""

from __future__ import annotations

from textwrap import dedent


LABELS = [
    "Background",
    "Method",
    "Result/Comparison",
    "Support",
    "Contrast/Criticism",
    "Perfunctory/Ceremonial",
]


def citation_intent_prompt(
    citing_title: str,
    citing_abstract: str,
    cited_title: str,
    cited_abstract: str,
) -> str:
    """Build a compact JSON-only prompt for title/abstract citation typing."""
    return dedent(
        f"""
        Classify the intent of a citation from the citing paper to the cited paper.

        Choose exactly one label:
        - Background: cited work provides real topical context or broad related work.
        - Method: citing work uses, extends, or depends on the cited method/tool/dataset.
        - Result/Comparison: citing work compares against, evaluates against, or discusses related results.
        - Support: cited work is evidence for a claim made by the citing work.
        - Contrast/Criticism: citing work disagrees with, corrects, or criticizes the cited work.
        - Perfunctory/Ceremonial: citation appears very weakly connected, generic authority-padding,
          ceremonial, or not substantively motivated by the title/abstract relationship.

        Important distinctions:
        - Use Method only when there is clear methodological/tool/dataset dependence.
        - Use Background for broad but genuinely relevant context.
        - Use Perfunctory/Ceremonial when the relationship is vague, weak, generic, or hard to justify
          from the two abstracts.
        - Use Contrast/Criticism only when the titles/abstracts indicate disagreement, correction,
          competing claims, or critique.

        Citing paper title:
        {citing_title}

        Citing paper abstract:
        {citing_abstract}

        Cited paper title:
        {cited_title}

        Cited paper abstract:
        {cited_abstract}

        Return only valid JSON:
        {{"label": "<one of {LABELS}>", "confidence": <number from 0 to 1>}}
        """
    ).strip()

