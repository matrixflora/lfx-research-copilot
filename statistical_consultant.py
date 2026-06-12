#!/usr/bin/env python3
"""
statistical_consultant.py — Assist with statistical planning by recommending
tests, estimating sample sizes, statistical power, and checking assumptions.

Outputs
-------
outputs/statistics/statistical_plan.md
outputs/statistics/sample_size_estimates.csv
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("statistical_consultant")

DESIGN_RECOMMENDATIONS = {
    "two_group_independent": {
        "recommended_test": "Independent samples t-test (parametric) or Mann-Whitney U (non-parametric)",
        "assumptions": ["Normality (Shapiro-Wilk)", "Homogeneity of variance (Levene's test)",
                        "Independence of observations"],
        "effect_size": "Cohen's d",
        "sample_size_formula": "n = 2 × ((Z_α/2 + Z_β) / (Δ / σ))²",
    },
    "two_group_paired": {
        "recommended_test": "Paired t-test (parametric) or Wilcoxon signed-rank (non-parametric)",
        "assumptions": ["Normality of differences", "No outliers in differences"],
        "effect_size": "Cohen's dz",
        "sample_size_formula": "n = ((Z_α/2 + Z_β) / (Δ / σ))²",
    },
    "multi_group": {
        "recommended_test": "One-way ANOVA (parametric) or Kruskal-Wallis (non-parametric)",
        "assumptions": ["Normality per group", "Homogeneity of variance",
                        "Independence of observations"],
        "effect_size": "η² or ω²",
        "sample_size_formula": "ANOVA power via G*Power or pingouin",
    },
    "correlation": {
        "recommended_test": "Pearson (parametric) or Spearman (non-parametric) correlation",
        "assumptions": ["Linearity", "Bivariate normality", "No outliers"],
        "effect_size": "r",
        "sample_size_formula": "n = ((Z_α/2 + Z_β) / 0.5 × ln((1+r)/(1-r)))² + 3",
    },
    "categorical": {
        "recommended_test": "Chi-square test of independence or Fisher's exact test",
        "assumptions": ["Expected frequency ≥5 per cell (chi-square)", "Independence"],
        "effect_size": "Cramér's V or φ",
        "sample_size_formula": "n = (Z_α/2 + Z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²",
    },
    "regression": {
        "recommended_test": "Multiple linear regression or logistic regression",
        "assumptions": ["Linearity", "Independence of residuals", "Homoscedasticity",
                        "Normality of residuals", "No multicollinearity (VIF < 5)"],
        "effect_size": "R² or f²",
        "sample_size_formula": "n = 10-20 × number of predictors (rule of thumb)",
    },
}


def recommend_statistical_tests(design: str = "two_group_independent",
                                n_groups: int = 2,
                                measurement: str = "continuous") -> Dict:
    key = design
    if key not in DESIGN_RECOMMENDATIONS:
        key = "two_group_independent"
    rec = dict(DESIGN_RECOMMENDATIONS[key])
    rec["design"] = design
    rec["measurement"] = measurement
    return rec


def estimate_sample_size(effect_size: float = 0.5,
                         power: float = 0.80,
                         alpha: float = 0.05,
                         design: str = "two_group_independent") -> Dict:
    """Simple sample size estimation using normal approximation."""
    Z_alpha = 1.96 if alpha == 0.05 else 2.576
    Z_beta = 0.84 if power == 0.80 else 1.28

    if effect_size <= 0:
        n = 100  # fallback
    else:
        if design in ("two_group_independent",):
            n = int(2 * ((Z_alpha + Z_beta) / effect_size) ** 2) + 1
        elif design in ("two_group_paired",):
            n = int(((Z_alpha + Z_beta) / effect_size) ** 2) + 1
        elif design in ("correlation",):
            n = int(((Z_alpha + Z_beta) / (0.5 * math.log((1 + effect_size) / (1 - effect_size)))) ** 2) + 3
        else:
            n = int(((Z_alpha + Z_beta) / effect_size) ** 2) + 1

    return {
        "design": design,
        "effect_size": effect_size,
        "power": power,
        "alpha": alpha,
        "estimated_n": max(n, 10),
        "interpretation": f"{max(n, 10)} participants needed ({design})",
    }


def estimate_power(n: int, effect_size: float = 0.5, alpha: float = 0.05) -> Dict:
    """Post-hoc power estimate."""
    Z_alpha = 1.96 if alpha == 0.05 else 2.576
    se = 2.0 / math.sqrt(n)
    Z_beta = abs(effect_size) / se - Z_alpha
    power_est = min(max(0.5 + 0.5 * math.erf(Z_beta / math.sqrt(2)), 0.05), 0.999)
    return {
        "n": n,
        "effect_size": effect_size,
        "alpha": alpha,
        "estimated_power": round(power_est, 3),
    }


def check_assumptions(data_desc: str = "continuous, two groups") -> List[Dict]:
    """Generate assumption checklist based on design description."""
    checks = []
    if "continuous" in data_desc:
        checks.append({"assumption": "Normality", "test": "Shapiro-Wilk", "heuristic": "p > 0.05"})
        checks.append({"assumption": "Homogeneity of variance", "test": "Levene's test", "heuristic": "p > 0.05"})
    if "correlation" in data_desc or "regression" in data_desc:
        checks.append({"assumption": "Linearity", "test": "Scatterplot / residual plot", "heuristic": "Visual inspection"})
        checks.append({"assumption": "Independence", "test": "Durbin-Watson", "heuristic": "1.5 < DW < 2.5"})
    if "categorical" in data_desc:
        checks.append({"assumption": "Expected frequencies", "test": "Chi-square minimum", "heuristic": "All cells ≥5"})
    return checks


def _generate_plan(recs: List[Dict], sizes: pd.DataFrame, power_est: Dict, checks: List[Dict]) -> str:
    lines: List[str] = []
    lines.append("# Statistical Plan\n")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Recommended Tests\n")
    for r in recs:
        lines.append(f"- **Design:** {r.get('design', 'unknown')}")
        lines.append(f"- **Test:** {r.get('recommended_test', '')}")
        lines.append(f"- **Effect size:** {r.get('effect_size', '')}")
        lines.append("- **Assumptions:**")
        for a in r.get("assumptions", []):
            lines.append(f"  - {a}")
        lines.append("")

    lines.append("## Sample Size Estimates\n")
    if not sizes.empty:
        lines.append(sizes.to_markdown(index=False))
        lines.append("")

    if power_est:
        lines.append("## Post-hoc Power\n")
        lines.append(f"- N = {power_est.get('n', '?')}, Effect = {power_est.get('effect_size', '?')}")
        lines.append(f"- Estimated power: {power_est.get('estimated_power', '?'):.1%}\n")

    lines.append("## Assumption Checks\n")
    for c in checks:
        lines.append(f"- **{c['assumption']}:** {c['test']} ({c['heuristic']})")

    lines.append("\n---\n*Generated by statistical_consultant.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical planning assistant.")
    parser.add_argument("--design", type=str, default="two_group_independent",
                        choices=list(DESIGN_RECOMMENDATIONS.keys()))
    parser.add_argument("--effect-size", type=float, default=0.5)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=None, help="Existing sample size for power estimate")
    parser.add_argument("--output-dir", type=str, default="outputs/statistics")
    args = parser.parse_args()

    rec = recommend_statistical_tests(args.design)
    size = estimate_sample_size(args.effect_size, args.power, args.alpha, args.design)
    sizes_df = pd.DataFrame([size])

    power_est = {}
    if args.n:
        power_est = estimate_power(args.n, args.effect_size, args.alpha)

    checks = check_assumptions(f"continuous, {args.design}")

    plan = _generate_plan([rec], sizes_df, power_est, checks)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "statistical_plan.md", "w") as f:
        f.write(plan)
    log.info("Saved -> %s", out_dir / "statistical_plan.md")

    sizes_df.to_csv(out_dir / "sample_size_estimates.csv", index=False)
    log.info("Saved -> %s", out_dir / "sample_size_estimates.csv")

    print(f"\n--- Statistical Consultant Complete ---")
    print()

if __name__ == "__main__":
    main()
