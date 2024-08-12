import math
import numpy as np


# Summation using numpy
def _sum(values):
    return np.sum(values)


# Compute mean of a list
def _mean(values):
    return _sum(values) / len(values)


# Compute variance of a list
def _variance(values):
    mean = _mean(values)
    return _sum([(v - mean) ** 2.0 for v in values]) / len(values)


# Compute d-prime for the given observations.
# Observations must be an array of (<label>,<score>) elements.
# Labels must be either 0 (impostor) or 1 (genuine).
# If either the number of impostors or genuine observations is zero, it returns 'NaN' as d-prime.
def compute_d_prime(observations):
    # Separate genuine and impostor scores
    genuine_scores = [score for label, score in observations if label == 1]
    impostor_scores = [score for label, score in observations if label == 0]

    # Check if there are enough observations for computation
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return float('NaN')

    # Compute mean and variance for genuine and impostor scores
    genuine_mean = _mean(genuine_scores)
    impostor_mean = _mean(impostor_scores)
    genuine_var = _variance(genuine_scores)
    impostor_var = _variance(impostor_scores)

    # Compute d-prime
    d_prime = math.sqrt(2.0) * abs(genuine_mean - impostor_mean) / math.sqrt(genuine_var + impostor_var)
    return d_prime


# Combine False Match Rate (FMR) and False Non-Match Rate (FNMR) computations for similarity and distance
# observations: Array of (<label>, <score>) elements
# threshold: Threshold for comparison
# comparison_type: 'similarity' or 'distance'
def compute_fmr_fnmr(observations, threshold, comparison_type='similarity'):
    impostor_count = 0
    genuine_count = 0
    false_match_count = 0
    false_non_match_count = 0

    # Iterate through observations and count impostors, genuines, false matches, and false non-matches
    for label, score in observations:
        if label == 0:  # Impostor observation
            impostor_count += 1
            if (comparison_type == 'similarity' and score >= threshold) or (
                    comparison_type == 'distance' and score <= threshold):
                false_match_count += 1
        else:  # Genuine observation
            genuine_count += 1
            if (comparison_type == 'similarity' and score < threshold) or (
                    comparison_type == 'distance' and score > threshold):
                false_non_match_count += 1

    # Calculate False Match Rate (FMR) and False Non-Match Rate (FNMR)
    fmr = false_match_count / impostor_count if impostor_count > 0 else float('NaN')
    fnmr = false_non_match_count / genuine_count if genuine_count > 0 else float('NaN')

    return fmr, fnmr


# Compute False Match Rate (FMR), False Non-Match Rate (FNMR), Equal Error Rate (EER), and Area Under the Curve (AUC)
# observations: Array of (<label>, <score>) elements
# comparison_type: 'similarity' or 'distance'
def compute_fmr_fnmr_eer_auc(observations, comparison_type='similarity'):
    scores = sorted(np.unique([score for _, score in observations]))

    if len(scores) == 0:
        return {
            "FMR": float('NaN'),
            "FNMR": float('NaN'),
            "EER_threshold": float('NaN'),
            "AUC": float('NaN'),
            "FMRS": None,
            "TMRS": None
        }

    output_fnmr = float('inf')
    output_fmr = float('inf')
    fnmr_fmr_diff = float('inf')
    output_threshold = float('inf')

    fmr = []
    tmr = []

    for threshold in scores:
        current_fmr, current_fnmr = compute_fmr_fnmr(observations, threshold, comparison_type)

        if not (float('-inf') < current_fmr < float('inf')) or not (float('-inf') < current_fnmr < float('inf')):
            return {
                "FMR": float('NaN'),
                "FNMR": float('NaN'),
                "EER_threshold": float('NaN'),
                "AUC": float('NaN'),
                "FMRS": None,
                "TMRS": None
            }

        current_diff = abs(current_fnmr - current_fmr)
        if current_diff <= fnmr_fmr_diff:
            output_fnmr = current_fnmr
            output_fmr = current_fmr
            fnmr_fmr_diff = current_diff
            output_threshold = threshold

        fmr.append(current_fmr)
        tmr.append(1.0 - current_fnmr)

    auc_parts = [abs(fmr[i] - fmr[i + 1]) * (tmr[i] + tmr[i + 1]) / 2.0 for i in range(len(fmr) - 1)]
    auc = _sum(auc_parts)

    return {
        "FMR": output_fmr,
        "FNMR": output_fnmr,
        "EER_threshold": output_threshold,
        "AUC": auc,
        "FMRS": fmr,
        "TMRS": tmr
    }


# # Example usage:
# observations = [(0, 0.1), (0, 0.4), (1, 0.35), (1, 0.8)]
# print(compute_d_prime(observations))
# print(compute_fmr_fnmr_eer_auc(observations, 'distance'))
