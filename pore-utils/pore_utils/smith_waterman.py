import numpy as np


def s_w(seq1, seq2, cost_fn={"match": 2, "mismatch": -2, "gap": -1}):
    '''Smith-Waterman implementation (also use this for TAing 427)'''
    nrows = len(seq1) + 1  # + 1 is to accommodate the 0th row
    ncols = len(seq2) + 1
    dp = np.zeros((nrows, ncols))

    def score():
        best_score = -np.Inf
        best_pos = (0, 0)
        for row_i in range(1, nrows):
            for col_j in range(1, ncols):
                score_ij = _score_ij(row_i, col_j)
                dp[row_i, col_j] = score_ij
                if score_ij >= best_score:
                    best_score = score_ij
                    best_pos = (row_i, col_j)
        return best_pos, best_score

    def _score_ij(i, j):
        if seq1[i - 1] == seq2[j - 1]:
            match_cost = cost_fn["match"]
        else:
            match_cost = cost_fn["mismatch"]

        up = dp[i - 1, j] + cost_fn["gap"]  # anything but diag must be a gap
        left = dp[i, j - 1] + cost_fn["gap"]
        dia = dp[i - 1, j - 1] + match_cost

        return max(up, left, dia)

    def traceback(best_pos):
        '''Ties prefer diagonal > up > left in this implementation'''
        seq1_out, seq2_out, match_str = "", "", ""
        row_i, col_j = best_pos

        # Deal with uneven sequences
        if row_i == col_j and row_i < nrows - 1 and col_j < ncols - 1:
            seq1_out = seq1[row_i:]
            seq2_out = seq2[col_j:]
            for i in range(row_i, nrows - 1):
                match_str += ":"
        if row_i != col_j and row_i < nrows - 1:
            seq1_out = seq1[row_i:]
            for i in range(row_i, nrows - 1):
                match_str += " "
                seq2_out += "-"
        if row_i != col_j and col_j < ncols - 1:
            seq2_out = seq2[col_j:]
            for i in range(col_j, ncols - 1):
                match_str += " "
                seq1_out += "-"

        # Traceback
        last_dia_s1 = 0
        last_dia_s2 = 0
        while row_i and col_j:  # end when either is 0
            up = dp[row_i - 1, col_j]
            left = dp[row_i, col_j - 1]
            dia = dp[row_i - 1, col_j - 1]

            # Case 1: diagonal
            if dia >= up and dia >= left:
                row_i -= 1
                col_j -= 1
                last_dia_s1 = row_i
                last_dia_s2 = col_j
                seq1_out = seq1[row_i] + seq1_out
                seq2_out = seq2[col_j] + seq2_out
                if seq1[row_i] == seq2[col_j]:
                    match_str = "|" + match_str
                else:
                    match_str = ":" + match_str
            # Case 2: up
            elif up >= left:
                row_i -= 1
                seq1_out = seq1[row_i] + seq1_out
                seq2_out = "-" + seq2_out
                match_str = " " + match_str
            # Case 3: left
            else:
                col_j -= 1
                seq1_out = "-" + seq1_out
                seq2_out = seq2[col_j] + seq2_out
                match_str = " " + match_str

        # Deal with uneven sequences
        if 0 < row_i:
            seq1_out = seq1[:row_i] + seq1_out
            for i in range(0, row_i):
                seq2_out = "-" + seq2_out
                match_str = " " + match_str
        if 0 < col_j:
            seq2_out = seq2[:col_j] + seq2_out
            for i in range(0, col_j):
                seq1_out = "-" + seq1_out
                match_str = " " + match_str

        return seq1_out, seq2_out, match_str, last_dia_s1, last_dia_s2

    best_pos, best_score = score()
    seq1_out, seq2_out, match_str, last_dia_s1, last_dia_s2 = traceback(
        best_pos)
    return best_pos, best_score, "\n".join([seq1_out, match_str, seq2_out]),\
        last_dia_s1, last_dia_s2
