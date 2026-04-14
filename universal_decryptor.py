from __future__ import annotations

import argparse
import heapq
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:-()'"
M = len(ALPHABET)
ALPHABET_INDEX = {ch: i for i, ch in enumerate(ALPHABET)}


# =========================================================
# БАЗА
# =========================================================

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)

def mod_inverse(a: int, m: int) -> int:
    a %= m
    if gcd(a, m) != 1:
        raise ValueError(f"No inverse for a={a} mod {m}")
    old_r, r = a, m
    old_s, s = 1, 0
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    return old_s % m

def is_supported_char(ch: str) -> bool:
    return ch in ALPHABET

def char_to_index(ch: str) -> int:
    return ALPHABET_INDEX[ch]

def index_to_char(i: int) -> str:
    return ALPHABET[i % M]

def supported_only(text: str) -> str:
    return "".join(ch for ch in text if ch in ALPHABET)


# =========================================================
# ЯЗЫКОВАЯ МОДЕЛЬ
# =========================================================

TRAINING_CORPUS = """
The morning sun rose slowly over the small town, painting the rooftops in shades of gold and orange.
Birds began to sing from the old oak tree near the town square. A gentle breeze carried the scent of
fresh bread from the local bakery. It was the kind of morning that made people want to get up early
and enjoy life. Maria, a young teacher who lived on Maple Street, opened her window and took a deep
breath. She loved mornings like this because they felt full of promise.

Reading books is a great way to relax and learn new things. It helps improve your vocabulary and
imagination. Moreover, it can reduce stress and make you feel better. Try to read a few pages every
day, even when you are busy. A short and useful habit often grows into a strong personal routine.

Cybersecurity students study networks, operating systems, Python scripts, and basic cryptography.
They learn how attackers think, how defenders monitor logs, and how analysts respond to incidents.
A careful engineer checks assumptions, verifies evidence, writes simple tools, and improves them step
by step. Good code is readable, testable, and honest about its limits.

On a quiet evening I sat near the window and decided to write a simple story about time, memory,
and choice. The street below was calm, the air was cool, and the city seemed to pause for a moment.
I thought about the strange way people change. At 18 you want speed, noise, risk, and proof that your
name will matter. At 25 you begin to value patience, skill, and the rare comfort of a clear plan.
By 30, perhaps, you learn that not every closed door is a loss.

The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
Natural English contains repeated words, predictable endings, common spaces, and punctuation patterns.
"""

ENGLISH_LETTER_FREQ = {
    "e": 12.70, "t": 9.06, "a": 8.17, "o": 7.51, "i": 6.97, "n": 6.75,
    "s": 6.33, "h": 6.09, "r": 5.99, "d": 4.25, "l": 4.03, "c": 2.78,
    "u": 2.76, "m": 2.41, "w": 2.36, "f": 2.23, "g": 2.02, "y": 1.97,
    "p": 1.93, "b": 1.49, "v": 0.98, "k": 0.77, "j": 0.15, "x": 0.15,
    "q": 0.10, "z": 0.07,
}

COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "time", "no", "just", "him", "know", "take", "people", "into", "year",
    "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "are", "was", "were", "has", "had", "been", "more", "very",
    "window", "street", "quiet", "story", "memory", "choice", "value", "plan",
    "progress", "small", "friend", "life", "built", "again",
}

def build_ngram_model(corpus: str, n: int) -> Tuple[Dict[str, float], float]:
    filtered = supported_only(corpus)
    counts = Counter(filtered[i:i+n] for i in range(len(filtered) - n + 1))
    total = sum(counts.values())
    vocab = max(1, len(counts))
    denom = total + vocab
    log_probs = {gram: math.log((cnt + 1) / denom) for gram, cnt in counts.items()}
    floor = math.log(1 / denom)
    return log_probs, floor

UNIGRAM_LOGP, UNIGRAM_FLOOR = build_ngram_model(TRAINING_CORPUS, 1)
BIGRAM_LOGP, BIGRAM_FLOOR = build_ngram_model(TRAINING_CORPUS, 2)
TRIGRAM_LOGP, TRIGRAM_FLOOR = build_ngram_model(TRAINING_CORPUS, 3)
TETRAGRAM_LOGP, TETRAGRAM_FLOOR = build_ngram_model(TRAINING_CORPUS, 4)

EXPECTED_CHAR_ORDER = [ch for ch, _ in Counter(supported_only(TRAINING_CORPUS)).most_common()]
for ch in ALPHABET:
    if ch not in EXPECTED_CHAR_ORDER:
        EXPECTED_CHAR_ORDER.append(ch)

def english_score(text: str) -> float:
    if not text:
        return float("-inf")

    filtered = supported_only(text)
    n = len(filtered)
    if n == 0:
        return float("-inf")

    score = 0.0

    if n >= 4:
        total = 0.0
        for i in range(n - 3):
            gram = filtered[i:i + 4]
            total += TETRAGRAM_LOGP.get(gram, TETRAGRAM_FLOOR)
        score += 2.6 * total / (n - 3)

    spaces = filtered.count(" ")
    letters = sum(ch.isalpha() for ch in filtered)
    punctuation = sum(ch in ".,!?;:-()'" for ch in filtered)

    score += 8.0 * spaces / n
    score += 4.5 * letters / n
    score += 0.8 * punctuation / n

    lowered = filtered.lower()
    letter_counts = Counter(ch for ch in lowered if "a" <= ch <= "z")
    total_letters = sum(letter_counts.values())
    if total_letters > 0:
        chi = 0.0
        for ch, expected_pct in ENGLISH_LETTER_FREQ.items():
            observed = letter_counts.get(ch, 0)
            expected = total_letters * expected_pct / 100.0
            if expected > 0:
                chi += (observed - expected) ** 2 / expected
        score += max(0.0, 150.0 - chi) * 0.08

    words = [tok for tok in re.split(r"[^A-Za-z]+", lowered) if tok]
    if words:
        common_hits = sum(1 for w in words if w in COMMON_WORDS)
        score += 2.2 * common_hits / max(1, len(words)) * 100.0

    score += 1.4 * lowered.count(" the ")
    score += 1.0 * lowered.count(" and ")
    score += 0.8 * lowered.count("ing")
    score += 0.8 * filtered.count(". ")
    score += 0.6 * filtered.count(", ")

    weird_pairs = ["  ", "..", ",,", "!!", "??", ";;", "::", ",.", ".,", "--", "''"]
    for pat in weird_pairs:
        score -= 1.2 * filtered.count(pat)

    return score


# =========================================================
# ПРОСТАЯ ЗАМЕНА
# =========================================================

@dataclass
class SubstitutionResult:
    plaintext: str
    key: str
    score: float

class SubstitutionDecryptor:
    def __init__(self, random_restarts: int = 40, iterations: int = 12000, seed: int = 42):
        self.random_restarts = random_restarts
        self.iterations = iterations
        self.random = random.Random(seed)

    def _cipher_indices_by_frequency(self, ciphertext: str) -> List[int]:
        counts = Counter(ch for ch in ciphertext if ch in ALPHABET)
        ordered = [ALPHABET_INDEX[ch] for ch, _ in counts.most_common()]
        for ch in ALPHABET:
            idx = ALPHABET_INDEX[ch]
            if idx not in ordered:
                ordered.append(idx)
        return ordered

    def _base_dec_map(self, ciphertext: str) -> List[str]:
        cipher_order = self._cipher_indices_by_frequency(ciphertext)
        dec_map = [""] * M
        used_plain = set()

        for cipher_idx, plain_ch in zip(cipher_order, EXPECTED_CHAR_ORDER):
            dec_map[cipher_idx] = plain_ch
            used_plain.add(plain_ch)

        leftovers = [ch for ch in ALPHABET if ch not in used_plain]
        for i in range(M):
            if not dec_map[i]:
                dec_map[i] = leftovers.pop(0)
        return dec_map

    def _decrypt_with_dec_map(self, text: str, dec_map: List[str]) -> str:
        return "".join(dec_map[ALPHABET_INDEX[ch]] if ch in ALPHABET else ch for ch in text)

    def _dec_map_to_encryption_key(self, dec_map: List[str]) -> str:
        plain_to_cipher = {}
        for cipher_idx, plain_ch in enumerate(dec_map):
            plain_to_cipher[plain_ch] = ALPHABET[cipher_idx]
        return "".join(plain_to_cipher[ch] for ch in ALPHABET)

    def _swap_values(self, dec_map: List[str], i: int, j: int) -> List[str]:
        new_map = dec_map[:]
        new_map[i], new_map[j] = new_map[j], new_map[i]
        return new_map

    def _perturb(self, dec_map: List[str], active_indices: List[int], swaps: int) -> List[str]:
        new_map = dec_map[:]
        if len(active_indices) < 2:
            return new_map
        for _ in range(swaps):
            i, j = self.random.sample(active_indices, 2)
            new_map[i], new_map[j] = new_map[j], new_map[i]
        return new_map

    def _space_forced_variants(self, base_map: List[str], ciphertext: str) -> List[List[str]]:
        counts = Counter(ch for ch in ciphertext if ch in ALPHABET)
        top_cipher = [ch for ch, _ in counts.most_common(8)]
        variants = [base_map]
        space_idx_current = next(i for i, ch in enumerate(base_map) if ch == " ")
        for ch in top_cipher[:4]:
            idx = ALPHABET_INDEX[ch]
            if idx == space_idx_current:
                continue
            v = base_map[:]
            v[idx], v[space_idx_current] = v[space_idx_current], v[idx]
            variants.append(v)
        return variants

    def decrypt(self, ciphertext: str) -> SubstitutionResult:
        active_indices = [ALPHABET_INDEX[ch] for ch in Counter(ch for ch in ciphertext if ch in ALPHABET).keys()]
        if len(active_indices) < 2:
            base_map = self._base_dec_map(ciphertext)
            plain = self._decrypt_with_dec_map(ciphertext, base_map)
            return SubstitutionResult(plain, self._dec_map_to_encryption_key(base_map), english_score(plain))

        base_map = self._base_dec_map(ciphertext)
        initial_variants = self._space_forced_variants(base_map, ciphertext)

        best_map = base_map
        best_plain = self._decrypt_with_dec_map(ciphertext, best_map)
        best_score = english_score(best_plain)

        total_restarts = max(self.random_restarts, len(initial_variants))
        for restart in range(total_restarts):
            if restart < len(initial_variants):
                current_map = initial_variants[restart][:]
            else:
                swaps = self.random.randint(8, 30)
                current_map = self._perturb(base_map, active_indices, swaps)

            current_plain = self._decrypt_with_dec_map(ciphertext, current_map)
            current_score = english_score(current_plain)

            local_best_map = current_map[:]
            local_best_plain = current_plain
            local_best_score = current_score

            no_improve = 0
            temperature = 4.5

            for _ in range(self.iterations):
                move_roll = self.random.random()

                if move_roll < 0.84:
                    i, j = self.random.sample(active_indices, 2)
                    new_map = self._swap_values(current_map, i, j)
                else:
                    k = self.random.randint(2, min(6, max(2, len(active_indices))))
                    picks = self.random.sample(active_indices, k)
                    new_map = current_map[:]
                    vals = [new_map[p] for p in picks]
                    self.random.shuffle(vals)
                    for pos, val in zip(picks, vals):
                        new_map[pos] = val

                new_plain = self._decrypt_with_dec_map(ciphertext, new_map)
                new_score = english_score(new_plain)
                delta = new_score - current_score

                accept = False
                if delta >= 0:
                    accept = True
                else:
                    prob = math.exp(delta / max(temperature, 1e-9))
                    if self.random.random() < prob:
                        accept = True

                if accept:
                    current_map = new_map
                    current_plain = new_plain
                    current_score = new_score
                    temperature *= 0.9995

                    if current_score > local_best_score:
                        local_best_map = current_map[:]
                        local_best_plain = current_plain
                        local_best_score = current_score
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

                if no_improve > 2600:
                    break

            if local_best_score > best_score:
                best_map = local_best_map
                best_plain = local_best_plain
                best_score = local_best_score

        improved = True
        while improved:
            improved = False
            for i in active_indices:
                for j in active_indices:
                    if i >= j:
                        continue
                    test_map = best_map[:]
                    test_map[i], test_map[j] = test_map[j], test_map[i]
                    test_plain = self._decrypt_with_dec_map(ciphertext, test_map)
                    test_score = english_score(test_plain)
                    if test_score > best_score:
                        best_map = test_map
                        best_plain = test_plain
                        best_score = test_score
                        improved = True

        return SubstitutionResult(
            plaintext=best_plain,
            key=self._dec_map_to_encryption_key(best_map),
            score=best_score,
        )


# =========================================================
# АФФИННЫЙ ШИФР
# =========================================================

@dataclass
class AffineResult:
    plaintext: str
    a: int
    b: int
    score: float

class AffineDecryptor:
    def _decrypt(self, text: str, a: int, b: int) -> str:
        a_inv = mod_inverse(a, M)
        out = []
        for ch in text:
            if ch in ALPHABET:
                y = char_to_index(ch)
                x = (a_inv * (y - b)) % M
                out.append(index_to_char(x))
            else:
                out.append(ch)
        return "".join(out)

    def decrypt(self, ciphertext: str) -> AffineResult:
        best = AffineResult(ciphertext, 1, 0, float("-inf"))
        for a in range(1, M):
            if gcd(a, M) != 1:
                continue
            for b in range(M):
                plain = self._decrypt(ciphertext, a, b)
                score = english_score(plain)
                if score > best.score:
                    best = AffineResult(plain, a, b, score)
        return best


# =========================================================
# АФФИННЫЙ РЕКУРРЕНТНЫЙ ШИФР
# Точно под main.py:
# a_i = a_(i-1) * a_(i-2) mod M
# b_i = b_(i-1) + b_(i-2) mod M
# =========================================================

@dataclass
class RecurrentResult:
    plaintext: str
    a1: int
    b1: int
    a2: int
    b2: int
    score: float

class RecurrentAffineDecryptor:
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def _decrypt_prefix_supported(self, text: str, a1: int, b1: int, a2: int, b2: int, limit: int) -> str:
        supported = [ch for ch in text if ch in ALPHABET][:limit]
        if not supported:
            return ""

        out = []
        prev2_a, prev2_b = a1 % M, b1 % M
        prev1_a, prev1_b = a2 % M, b2 % M

        for idx, ch in enumerate(supported):
            if idx == 0:
                a, b = prev2_a, prev2_b
            elif idx == 1:
                a, b = prev1_a, prev1_b
            else:
                a = (prev1_a * prev2_a) % M
                b = (prev1_b + prev2_b) % M
                prev2_a, prev2_b = prev1_a, prev1_b
                prev1_a, prev1_b = a, b

            if a == 0:
                return ""
            inv = mod_inverse(a, M)
            y = char_to_index(ch)
            x = (inv * (y - b)) % M
            out.append(index_to_char(x))

        return "".join(out)

    def _decrypt_full(self, text: str, a1: int, b1: int, a2: int, b2: int) -> str:
        out = []
        prev2_a, prev2_b = a1 % M, b1 % M
        prev1_a, prev1_b = a2 % M, b2 % M
        supported_index = 0

        for ch in text:
            if ch not in ALPHABET:
                out.append(ch)
                continue

            if supported_index == 0:
                a, b = prev2_a, prev2_b
            elif supported_index == 1:
                a, b = prev1_a, prev1_b
            else:
                a = (prev1_a * prev2_a) % M
                b = (prev1_b + prev2_b) % M
                prev2_a, prev2_b = prev1_a, prev1_b
                prev1_a, prev1_b = a, b

            if a == 0:
                out.append("?")
            else:
                inv = mod_inverse(a, M)
                y = char_to_index(ch)
                x = (inv * (y - b)) % M
                out.append(index_to_char(x))

            supported_index += 1

        return "".join(out)

    def _candidate_score(self, ciphertext: str, a1: int, b1: int, a2: int, b2: int, prefix_len: int) -> float:
        if a1 % M == 0 or a2 % M == 0:
            return float("-inf")
        plain = self._decrypt_prefix_supported(ciphertext, a1, b1, a2, b2, prefix_len)
        if not plain:
            return float("-inf")
        return english_score(plain)

    def _push_top(self, heap: List[Tuple[float, Tuple[int, int, int, int]]], score: float, cand: Tuple[int, int, int, int], limit: int):
        if len(heap) < limit:
            heapq.heappush(heap, (score, cand))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, cand))

    def _neighbors(self, a1: int, b1: int, a2: int, b2: int) -> List[Tuple[int, int, int, int]]:
        vals = []
        deltas = [-3, -2, -1, 1, 2, 3]
        for d in deltas:
            na1 = (a1 + d) % M
            if na1 != 0:
                vals.append((na1, b1, a2, b2))
            na2 = (a2 + d) % M
            if na2 != 0:
                vals.append((a1, b1, na2, b2))
            vals.append((a1, (b1 + d) % M, a2, b2))
            vals.append((a1, b1, a2, (b2 + d) % M))
        return vals

    def decrypt(self, ciphertext: str) -> RecurrentResult:
        supported = [ch for ch in ciphertext if ch in ALPHABET]
        if len(supported) < 2:
            return RecurrentResult(ciphertext, 1, 0, 1, 0, float("-inf"))

        y1 = char_to_index(supported[0])
        y2 = char_to_index(supported[1])

        # Берем самые вероятные символы начала текста
        plain_candidates = EXPECTED_CHAR_ORDER[:20]
        plain_idx = [char_to_index(ch) for ch in plain_candidates]

        prefix_len = min(80, len(supported))
        coarse_heap: List[Tuple[float, Tuple[int, int, int, int]]] = []

        # Этап 1. По первым двум символам вычисляем b1 и b2 из гипотез по x1,x2,
        # перебираем a1,a2 и быстро оцениваем префикс.
        for a1 in range(1, M):
            for x1 in plain_idx:
                b1 = (y1 - a1 * x1) % M
                for a2 in range(1, M):
                    for x2 in plain_idx:
                        b2 = (y2 - a2 * x2) % M
                        score = self._candidate_score(ciphertext, a1, b1, a2, b2, prefix_len)
                        self._push_top(coarse_heap, score, (a1, b1, a2, b2), limit=250)

        top_candidates = [cand for _, cand in sorted(coarse_heap, reverse=True)]

        # Этап 2. Локальное улучшение на полном тексте
        best = RecurrentResult(ciphertext, 1, 0, 1, 0, float("-inf"))
        visited = set()

        for a1, b1, a2, b2 in top_candidates:
            current = (a1, b1, a2, b2)
            current_plain = self._decrypt_full(ciphertext, *current)
            current_score = english_score(current_plain)

            improved = True
            while improved:
                improved = False
                for neigh in self._neighbors(*current):
                    if neigh in visited:
                        continue
                    visited.add(neigh)
                    plain = self._decrypt_full(ciphertext, *neigh)
                    score = english_score(plain)
                    if score > current_score:
                        current = neigh
                        current_plain = plain
                        current_score = score
                        improved = True

            if current_score > best.score:
                best = RecurrentResult(current_plain, current[0], current[1], current[2], current[3], current_score)

        return best


# =========================================================
# AUTO MODE
# =========================================================

def auto_decrypt(ciphertext: str, restarts: int, iterations: int) -> Tuple[str, str, float]:
    affine = AffineDecryptor().decrypt(ciphertext)
    subst = SubstitutionDecryptor(random_restarts=restarts, iterations=iterations).decrypt(ciphertext)
    recur = RecurrentAffineDecryptor().decrypt(ciphertext)

    variants = [
        ("affine", affine.plaintext, affine.score, f"a={affine.a}, b={affine.b}"),
        ("substitution", subst.plaintext, subst.score, f"key={subst.key}"),
        ("recurrent", recur.plaintext, recur.score, f"a1={recur.a1}, b1={recur.b1}, a2={recur.a2}, b2={recur.b2}"),
    ]
    best = max(variants, key=lambda x: x[2])
    return best


# =========================================================
# CLI
# =========================================================

def read_text(args) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("Нужно передать --text или --file")

def main():
    parser = argparse.ArgumentParser(description="Практический дешифратор для substitution / affine / recurrent")
    parser.add_argument("cipher", choices=["substitution", "affine", "recurrent", "auto"])
    parser.add_argument("--file", type=str)
    parser.add_argument("--text", type=str)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=12000)
    args = parser.parse_args()

    ciphertext = read_text(args)

    if args.cipher == "substitution":
        result = SubstitutionDecryptor(
            random_restarts=args.restarts,
            iterations=args.iterations,
        ).decrypt(ciphertext)

        print("=" * 60)
        print("SUBSTITUTION")
        print("=" * 60)
        print(f"Score: {result.score:.3f}")
        print("Recovered key:")
        print(result.key)
        print("\nPlaintext:\n")
        print(result.plaintext)

    elif args.cipher == "affine":
        result = AffineDecryptor().decrypt(ciphertext)

        print("=" * 60)
        print("AFFINE")
        print("=" * 60)
        print(f"Score: {result.score:.3f}")
        print(f"Recovered key: a={result.a}, b={result.b}")
        print("\nPlaintext:\n")
        print(result.plaintext)

    elif args.cipher == "recurrent":
        result = RecurrentAffineDecryptor().decrypt(ciphertext)

        print("=" * 60)
        print("RECURRENT AFFINE")
        print("=" * 60)
        print(f"Score: {result.score:.3f}")
        print(f"Recovered key: a1={result.a1}, b1={result.b1}, a2={result.a2}, b2={result.b2}")
        print("\nPlaintext:\n")
        print(result.plaintext)

    else:
        cipher_name, plaintext, score, params = auto_decrypt(
            ciphertext,
            restarts=args.restarts,
            iterations=args.iterations,
        )
        print("=" * 60)
        print("AUTO")
        print("=" * 60)
        print(f"Detected: {cipher_name}")
        print(f"Score: {score:.3f}")
        print(f"Params: {params}")
        print("\nPlaintext:\n")
        print(plaintext)

if __name__ == "__main__":
    main()