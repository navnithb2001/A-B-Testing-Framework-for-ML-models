"""
Statistical tests for A/B testing
"""
import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any
import math


def welch_t_test(group_a: List[float], group_b: List[float]) -> Dict[str, float]:
    """
    Perform Welch's t-test (doesn't assume equal variance)
    Used for comparing continuous metrics between two groups
    
    Returns:
        - t_statistic: The t-test statistic
        - p_value: Probability of observing this difference by chance
        - degrees_of_freedom: Degrees of freedom for the test
    """
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("Need at least 2 samples in each group")
    
    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    # Calculate degrees of freedom for Welch's test
    n1, n2 = len(group_a), len(group_b)
    s1, s2 = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    
    df = ((s1/n1 + s2/n2)**2) / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": float(df)
    }


def chi_square_test(contingency_table: np.ndarray) -> Dict[str, float]:
    """
    Perform chi-square test for categorical data
    Used for comparing proportions (e.g., conversion rates)
    
    Args:
        contingency_table: 2x2 array like [[successes_a, failures_a], 
                                           [successes_b, failures_b]]
    
    Returns:
        - chi2_statistic: The chi-square statistic
        - p_value: Probability of observing this difference by chance
        - degrees_of_freedom: Degrees of freedom (always 1 for 2x2)
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "expected_frequencies": expected.tolist()
    }


def proportions_z_test(successes_a: int, n_a: int, 
                       successes_b: int, n_b: int) -> Dict[str, float]:
    """
    Z-test for comparing two proportions
    Alternative to chi-square for conversion rate comparisons
    
    Args:
        successes_a: Number of successes in group A
        n_a: Total trials in group A
        successes_b: Number of successes in group B
        n_b: Total trials in group B
    """
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    
    # Pooled proportion
    p_pooled = (successes_a + successes_b) / (n_a + n_b)
    
    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    
    # Z-statistic
    z_stat = (p_a - p_b) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
        "proportion_a": float(p_a),
        "proportion_b": float(p_b),
        "pooled_proportion": float(p_pooled)
    }


def confidence_interval(data: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for a sample mean
    
    Args:
        data: List of values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        - mean: Sample mean
        - lower_bound: Lower bound of CI
        - upper_bound: Upper bound of CI
        - margin_of_error: Half-width of CI
    """
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 samples")
    
    mean = np.mean(data)
    std_err = stats.sem(data)
    
    # t-distribution for small samples
    df = n - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    
    margin_of_error = t_critical * std_err
    
    return {
        "mean": float(mean),
        "lower_bound": float(mean - margin_of_error),
        "upper_bound": float(mean + margin_of_error),
        "margin_of_error": float(margin_of_error),
        "confidence_level": confidence
    }


def confidence_interval_difference(group_a: List[float], group_b: List[float], 
                                   confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for the difference between two means
    
    Returns CI for (mean_b - mean_a)
    """
    n_a, n_b = len(group_a), len(group_b)
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    
    # Standard error of the difference
    se_diff = math.sqrt(var_a/n_a + var_b/n_b)
    
    # Degrees of freedom (Welch-Satterthwaite)
    df = ((var_a/n_a + var_b/n_b)**2) / ((var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1))
    
    # t-critical value
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    
    difference = mean_b - mean_a
    margin_of_error = t_critical * se_diff
    
    return {
        "difference": float(difference),
        "lower_bound": float(difference - margin_of_error),
        "upper_bound": float(difference + margin_of_error),
        "margin_of_error": float(margin_of_error),
        "confidence_level": confidence
    }


def effect_size_cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """
    Calculate Cohen's d effect size
    Standardized measure of the difference between two groups
    
    Interpretation:
        - 0.2: Small effect
        - 0.5: Medium effect
        - 0.8: Large effect
    """
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    n_a, n_b = len(group_a), len(group_b)
    
    # Pooled standard deviation
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    cohens_d = (mean_b - mean_a) / pooled_std
    
    return float(cohens_d)


def sample_size_calculator(baseline_rate: float, 
                           minimum_detectable_effect: float,
                           alpha: float = 0.05, 
                           power: float = 0.80) -> int:
    """
    Calculate required sample size per variant for A/B test
    
    Args:
        baseline_rate: Current conversion rate (e.g., 0.10 for 10%)
        minimum_detectable_effect: Relative change to detect (e.g., 0.20 for 20% improvement)
        alpha: Significance level (Type I error rate)
        power: Statistical power (1 - Type II error rate)
    
    Returns:
        Required sample size per variant
    """
    # New rate after improvement
    new_rate = baseline_rate * (1 + minimum_detectable_effect)
    
    # Effect size (Cohen's h for proportions)
    h = 2 * (math.asin(math.sqrt(new_rate)) - math.asin(math.sqrt(baseline_rate)))
    
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = stats.norm.ppf(power)
    
    # Sample size per group
    n = ((z_alpha + z_beta) / h) ** 2
    
    return math.ceil(n)


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Bonferroni correction for multiple comparisons
    
    Adjusts p-values when testing multiple hypotheses to control
    family-wise error rate
    
    Args:
        p_values: List of original p-values
    
    Returns:
        List of adjusted p-values
    """
    n_tests = len(p_values)
    adjusted = [min(p * n_tests, 1.0) for p in p_values]
    return adjusted


def is_statistically_significant(p_value: float, alpha: float = 0.05) -> bool:
    """
    Determine if result is statistically significant
    
    Args:
        p_value: P-value from statistical test
        alpha: Significance level (default 0.05 for 95% confidence)
    
    Returns:
        True if p_value < alpha (statistically significant)
    """
    return p_value < alpha
