import argparse
import math
import random
from collections import Counter

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:-()'"
M = len(ALPHABET)

# ============================================================
# ===================== LANGUAGE SCORE ========================
# ============================================================

COMMON_BIGRAMS = [
    "th","he","in","er","an","re","on","at","en","nd",
    "ti","es","or","te","of","ed","is","it","al","ar",
    "st","to","nt","ng","se","ha","as","ou","io","le"
]

COMMON_TRIGRAMS = [
    "the","and","ing","ent","ion","her","for","tha","nth",
    "int","ere","tio","ter","est","ers","ati","all"
]

COMMON_WORDS = {
    "the","be","to","of","and","a","in","that","have","i","it","for",
    "not","on","with","he","as","you","do","at","this","but","his",
    "by","from","they","we","say","her","she","or","an","will","my",
    "one","all","would","there","their","what","so","up","out","if"
}

def english_score(text):
    if not text:
        return -1e9

    score = 0.0
    lowered = text.lower()

    # буквенная плотность
    letters = sum(c.isalpha() for c in text)
    spaces = text.count(" ")
    score += 3.0 * letters / len(text)
    score += 2.0 * spaces / len(text)

    # биграммы
    for bg in COMMON_BIGRAMS:
        score += lowered.count(bg) * 1.2

    # триграммы
    for tg in COMMON_TRIGRAMS:
        score += lowered.count(tg) * 2.5

    # частые слова
    words = ''.join(c if c.isalpha() else ' ' for c in lowered).split()
    for w in words:
        if w in COMMON_WORDS:
            score += 3.0

    # штраф за мусор
    bad = sum(1 for i in range(len(text)-2)
              if text[i].isdigit() and text[i+1].isdigit() and text[i+2].isdigit())
    score -= bad * 0.8

    return score

# ============================================================
# =================== SUBSTITUTION ============================
# ============================================================

class SubstitutionDecryptor:

    def __init__(self, restarts=300, iterations=20000, seed=42):
        self.restarts = restarts
        self.iterations = iterations
        self.random = random.Random(seed)

    def decrypt_with_key(self, text, key):
        reverse = {key[i]: ALPHABET[i] for i in range(M)}
        return ''.join(reverse.get(c, c) for c in text)

    def initial_key(self, ciphertext):
        counts = Counter(ciphertext)
        cipher_sorted = [c for c,_ in counts.most_common()]
        for c in ALPHABET:
            if c not in cipher_sorted:
                cipher_sorted.append(c)

        expected = list(" etaoinshrdlcumwfgypbvkjxqzETAOINSHRDLUCMWFGYPBVKJXQZ0123456789.,!?;:-()'")
        reverse_map = dict(zip(cipher_sorted, expected))

        plain_to_cipher = {p:c for c,p in reverse_map.items()}

        key=[]
        used=set()
        for p in ALPHABET:
            c = plain_to_cipher.get(p)
            if not c or c in used:
                for cand in ALPHABET:
                    if cand not in used:
                        c=cand
                        break
            key.append(c)
            used.add(c)

        return ''.join(key)

    def mutate(self, key):
        arr=list(key)
        r=self.random.random()
        if r<0.8:
            i,j=self.random.sample(range(M),2)
            arr[i],arr[j]=arr[j],arr[i]
        else:
            i,j,k=self.random.sample(range(M),3)
            arr[i],arr[j],arr[k]=arr[j],arr[k],arr[i]
        return ''.join(arr)

    def decrypt(self, ciphertext):

        best_key = self.initial_key(ciphertext)
        best_plain = self.decrypt_with_key(ciphertext, best_key)
        best_score = english_score(best_plain)

        for _ in range(self.restarts):

            key = self.mutate(best_key)
            plain = self.decrypt_with_key(ciphertext,key)
            score = english_score(plain)

            temperature = 4.0

            for _ in range(self.iterations):
                new_key = self.mutate(key)
                new_plain = self.decrypt_with_key(ciphertext,new_key)
                new_score = english_score(new_plain)

                delta = new_score-score

                if delta>0 or self.random.random()<math.exp(delta/temperature):
                    key,plain,score=new_key,new_plain,new_score
                    temperature*=0.9995

                    if score>best_score:
                        best_key,best_plain,best_score=key,plain,score

        # финальный hill climb
        improved=True
        while improved:
            improved=False
            for i in range(M):
                for j in range(i+1,M):
                    arr=list(best_key)
                    arr[i],arr[j]=arr[j],arr[i]
                    test_key=''.join(arr)
                    test_plain=self.decrypt_with_key(ciphertext,test_key)
                    test_score=english_score(test_plain)
                    if test_score>best_score:
                        best_key,best_plain,best_score=test_key,test_plain,test_score
                        improved=True

        return best_plain,best_key,best_score

# ============================================================
# ================= RECURRENT ================================
# ============================================================

def gcd(a,b):
    while b:
        a,b=b,a%b
    return abs(a)

def modinv(a,m):
    for i in range(1,m):
        if (a*i)%m==1:
            return i
    raise ValueError()

def generate_keys(length,a1,b1,a2,b2):
    keys=[(a1,b1),(a2,b2)]
    while len(keys)<length:
        a=(keys[-1][0]*keys[-2][0])%M
        b=(keys[-1][1]+keys[-2][1])%M
        keys.append((a,b))
    return keys

def decrypt_recurrent(ciphertext):

    valid_a=[a for a in range(M) if gcd(a,M)==1]
    best_score=-1e9
    best_plain=ciphertext

    # beam search по a1,a2
    for a1 in valid_a:
        for a2 in valid_a:

            for b1 in range(0,M,5):
                for b2 in range(0,M,5):

                    try:
                        keys=generate_keys(len(ciphertext),a1,b1,a2,b2)
                        plain=[]
                        for i,c in enumerate(ciphertext):
                            if c in ALPHABET:
                                y=ALPHABET.index(c)
                                a,b=keys[i]
                                inv=modinv(a,M)
                                x=(inv*(y-b))%M
                                plain.append(ALPHABET[x])
                            else:
                                plain.append(c)

                        text=''.join(plain)
                        score=english_score(text)

                        if score>best_score:
                            best_score=score
                            best_plain=text
                    except:
                        continue

    return best_plain,best_score

# ============================================================
# ================= CLI ======================================
# ============================================================

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("cipher",choices=["substitution","recurrent"])
    parser.add_argument("--file",required=True)
    parser.add_argument("--restarts",type=int,default=300)
    parser.add_argument("--iterations",type=int,default=20000)
    args=parser.parse_args()

    with open(args.file,"r",encoding="utf-8") as f:
        ciphertext=f.read().strip()

    if args.cipher=="substitution":
        dec=SubstitutionDecryptor(args.restarts,args.iterations)
        plain,key,score=dec.decrypt(ciphertext)
        print("Score:",score)
        print("Key:",key)
        print("\nDecrypted:\n")
        print(plain)
    else:
        plain,score=decrypt_recurrent(ciphertext)
        print("Score:",score)
        print("\nDecrypted:\n")
        print(plain)

if __name__=="__main__":
    main()
    