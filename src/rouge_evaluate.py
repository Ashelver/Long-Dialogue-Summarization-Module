import json

def compute_rouge_n(reference, candidate, n):
    """
    Calculate ROUGE-N scores (including Precision, Recall and F-measure)
    """
    # Tokenize the text
    ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]
    cand_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)]
    
    ref_ngram_count = len(ref_ngrams)
    cand_ngram_count = len(cand_ngrams)
    
    # Number of matching n-grams
    overlapping_ngrams = set(ref_ngrams) & set(cand_ngrams)
    overlap_count = sum(min(ref_ngrams.count(ng), cand_ngrams.count(ng)) for ng in overlapping_ngrams)

    # Calculate Precision, Recall and F-measure
    precision = overlap_count / cand_ngram_count if cand_ngram_count > 0 else 0
    recall = overlap_count / ref_ngram_count if ref_ngram_count > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure


def compute_rouge_l(reference, candidate):
    """
    Calculate ROUGE-L scores (including Precision, Recall and F-measure)
    """
    # Dynamic programming to solve the longest common subsequence length (LCS)
    m, n = len(reference), len(candidate)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == candidate[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_length = dp[m][n]

    # Calculate Precision, Recall and F-measure
    precision = lcs_length / n if n > 0 else 0
    recall = lcs_length / m if m > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure




def print_rouge(name, json_path):
    list = []
    with open(json_path, 'r', encoding='utf-8') as file:
        list = json.load(file)

    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
    count = 0

    for entry in list:
        output = entry['output'].split()
        target = entry['target'].split()

        # ROUGE-1
        _, _, rouge1_f = compute_rouge_n(target, output, n=1)
        # ROUGE-2
        _, _, rouge2_f = compute_rouge_n(target, output, n=2)
        # ROUGE-L
        _, _, rougeL_f = compute_rouge_l(target, output)

        total_rouge1 += rouge1_f
        total_rouge2 += rouge2_f
        total_rougeL += rougeL_f
        count += 1

    # Average
    avg_rouge1 = total_rouge1 / count if count > 0 else 0
    avg_rouge2 = total_rouge2 / count if count > 0 else 0
    avg_rougeL = total_rougeL / count if count > 0 else 0

    # print
    print(name, f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(name, f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(name, f"Average ROUGE-L: {avg_rougeL:.4f}")



if __name__ == '__main__':
    print_rouge('AMI','../results/ami.json')
    print_rouge('ICSI','../results/icsi.json')
