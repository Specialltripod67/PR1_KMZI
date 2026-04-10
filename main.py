from typing import List, Tuple

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:-()'"
M = len(ALPHABET)


# =========================================================
# 1. Математические функции
# =========================================================

def gcd(a: int, b: int) -> int:
    """Наибольший общий делитель."""
    while b != 0:
        a, b = b, a % b
    return abs(a)


def mod_inverse(a: int, m: int) -> int:
    """
    Обратный элемент a^(-1) по модулю m.
    Бросает ValueError, если обратного элемента не существует.
    """
    a = a % m
    if gcd(a, m) != 1:
        raise ValueError(f"Обратный элемент для a={a} по модулю {m} не существует.")

    old_r, r = a, m
    old_s, s = 1, 0

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s

    return old_s % m


def is_valid_affine_multiplier(a: int, m: int = M) -> bool:
    """Проверка корректности множителя a для аффинного шифра."""
    return gcd(a % m, m) == 1


# =========================================================
# 2. Функции для алфавита
# =========================================================

def is_supported_char(ch: str) -> bool:
    return ch in ALPHABET


def char_to_index(ch: str) -> int:
    return ALPHABET.index(ch)


def index_to_char(index: int) -> str:
    return ALPHABET[index % M]


def validate_substitution_key(key: str) -> Tuple[bool, str]:
    """
    Проверяет ключ простой замены.
    Ключ должен быть перестановкой ALPHABET.
    """
    if len(key) != M:
        return False, f"Длина ключа должна быть {M}, сейчас {len(key)}."

    if set(key) != set(ALPHABET):
        missing = sorted(set(ALPHABET) - set(key))
        extra = sorted(set(key) - set(ALPHABET))
        message_parts = ["Ключ должен содержать ровно те же символы, что и ALPHABET."]
        if missing:
            message_parts.append(f"Отсутствуют символы: {''.join(missing)}")
        if extra:
            message_parts.append(f"Лишние символы: {''.join(extra)}")
        return False, " ".join(message_parts)

    if len(set(key)) != M:
        return False, "В ключе есть повторяющиеся символы."

    return True, "Ключ корректен."


# =========================================================
# 3. Шифр простой замены
# =========================================================

def encrypt_substitution(text: str, key: str) -> str:
    is_valid, message = validate_substitution_key(key)
    if not is_valid:
        raise ValueError(message)

    mapping = {ALPHABET[i]: key[i] for i in range(M)}
    result = []

    for ch in text:
        result.append(mapping[ch] if ch in mapping else ch)

    return "".join(result)


def decrypt_substitution(text: str, key: str) -> str:
    is_valid, message = validate_substitution_key(key)
    if not is_valid:
        raise ValueError(message)

    reverse_mapping = {key[i]: ALPHABET[i] for i in range(M)}
    result = []

    for ch in text:
        result.append(reverse_mapping[ch] if ch in reverse_mapping else ch)

    return "".join(result)


# =========================================================
# 4. Аффинный шифр
# =========================================================

def normalize_affine_key(a: int, b: int, m: int = M) -> Tuple[int, int]:
    a = a % m
    b = b % m
    return a, b


def encrypt_affine(text: str, a: int, b: int) -> str:
    a, b = normalize_affine_key(a, b)

    if not is_valid_affine_multiplier(a, M):
        raise ValueError(
            f"Некорректный ключ: a={a}. Нужно, чтобы gcd(a, {M}) = 1."
        )

    result = []

    for ch in text:
        if is_supported_char(ch):
            x = char_to_index(ch)
            y = (a * x + b) % M
            result.append(index_to_char(y))
        else:
            result.append(ch)

    return "".join(result)


def decrypt_affine(text: str, a: int, b: int) -> str:
    a, b = normalize_affine_key(a, b)

    if not is_valid_affine_multiplier(a, M):
        raise ValueError(
            f"Некорректный ключ: a={a}. Нужно, чтобы gcd(a, {M}) = 1."
        )

    a_inv = mod_inverse(a, M)
    result = []

    for ch in text:
        if is_supported_char(ch):
            y = char_to_index(ch)
            x = (a_inv * (y - b)) % M
            result.append(index_to_char(x))
        else:
            result.append(ch)

    return "".join(result)


# =========================================================
# 5. Аффинный рекуррентный шифр
# =========================================================

def generate_recurrent_keys(length: int, a1: int, b1: int, a2: int, b2: int) -> List[Tuple[int, int]]:
    """
    Генерирует последовательность ключей длины length:
    k1 = (a1, b1)
    k2 = (a2, b2)
    далее:
      a_i = (a_{i-1} * a_{i-2}) mod M
      b_i = (b_{i-1} + b_{i-2}) mod M
    """
    if length <= 0:
        return []

    a1, b1 = normalize_affine_key(a1, b1)
    a2, b2 = normalize_affine_key(a2, b2)

    if not is_valid_affine_multiplier(a1, M):
        raise ValueError(f"Некорректный множитель a1={a1}. Нужно, чтобы gcd(a1, {M}) = 1.")
    if not is_valid_affine_multiplier(a2, M):
        raise ValueError(f"Некорректный множитель a2={a2}. Нужно, чтобы gcd(a2, {M}) = 1.")

    keys = [(a1, b1)]

    if length == 1:
        return keys

    keys.append((a2, b2))

    while len(keys) < length:
        prev2_a, prev2_b = keys[-2]
        prev1_a, prev1_b = keys[-1]

        next_a = (prev1_a * prev2_a) % M
        next_b = (prev1_b + prev2_b) % M

        if not is_valid_affine_multiplier(next_a, M):
            raise ValueError(
                f"Сгенерирован некорректный множитель a={next_a} в рекуррентном ключе."
            )

        keys.append((next_a, next_b))

    return keys


def encrypt_recurrent_affine(text: str, a1: int, b1: int, a2: int, b2: int) -> str:
    """
    Шифрование аффинным рекуррентным шифром.
    Ключевой поток генерируется по длине всей строки.
    Символы вне ALPHABET оставляются без изменений,
    но их позиция учитывается в потоке ключей.
    """
    keys = generate_recurrent_keys(len(text), a1, b1, a2, b2)
    result = []

    for i, ch in enumerate(text):
        a, b = keys[i]
        if is_supported_char(ch):
            x = char_to_index(ch)
            y = (a * x + b) % M
            result.append(index_to_char(y))
        else:
            result.append(ch)

    return "".join(result)


def decrypt_recurrent_affine(text: str, a1: int, b1: int, a2: int, b2: int) -> str:
    """
    Расшифрование аффинным рекуррентным шифром.
    Генерируется тот же поток ключей, что и при шифровании.
    """
    keys = generate_recurrent_keys(len(text), a1, b1, a2, b2)
    result = []

    for i, ch in enumerate(text):
        a, b = keys[i]
        if is_supported_char(ch):
            y = char_to_index(ch)
            a_inv = mod_inverse(a, M)
            x = (a_inv * (y - b)) % M
            result.append(index_to_char(x))
        else:
            result.append(ch)

    return "".join(result)


# =========================================================
# 6. Ввод ключей
# =========================================================

def input_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            return int(raw)
        except ValueError:
            print("Ошибка: нужно ввести целое число.")


def input_substitution_key() -> str:
    print("\nВведите ключ простой замены.")
    print(f"Ключ должен быть перестановкой всех {M} символов ALPHABET.")
    print("Текущий ALPHABET:")
    print(ALPHABET)

    while True:
        key = input("Ключ: ")
        is_valid, message = validate_substitution_key(key)
        if is_valid:
            return key
        print(f"Ошибка: {message}")


def input_affine_key() -> Tuple[int, int]:
    print("\nВведите ключ для аффинного шифра.")
    print(f"Все вычисления идут по модулю M = {M}.")

    while True:
        a = input_int("Введите a: ")
        b = input_int("Введите b: ")

        a, b = normalize_affine_key(a, b)

        if not is_valid_affine_multiplier(a, M):
            print(f"Ошибка: a={a} некорректно. Нужно, чтобы gcd(a, {M}) = 1.")
            continue

        print(f"Нормализованный ключ: a={a}, b={b}")
        return a, b


def input_recurrent_affine_key() -> Tuple[int, int, int, int]:
    print("\nВведите две стартовые пары для аффинного рекуррентного шифра.")
    print(f"Все вычисления идут по модулю M = {M}.")

    while True:
        a1 = input_int("Введите a1: ")
        b1 = input_int("Введите b1: ")
        a2 = input_int("Введите a2: ")
        b2 = input_int("Введите b2: ")

        a1, b1 = normalize_affine_key(a1, b1)
        a2, b2 = normalize_affine_key(a2, b2)

        if not is_valid_affine_multiplier(a1, M):
            print(f"Ошибка: a1={a1} некорректно. Нужно, чтобы gcd(a1, {M}) = 1.")
            continue

        if not is_valid_affine_multiplier(a2, M):
            print(f"Ошибка: a2={a2} некорректно. Нужно, чтобы gcd(a2, {M}) = 1.")
            continue

        print(f"Нормализованные ключи: (a1, b1)=({a1}, {b1}), (a2, b2)=({a2}, {b2})")
        return a1, b1, a2, b2


# =========================================================
# 7. Интерфейс
# =========================================================

def choose_cipher() -> int:
    print("\nВыберите шифр:")
    print("1 - Шифр простой замены")
    print("2 - Аффинный шифр")
    print("3 - Аффинный рекуррентный шифр")
    print("0 - Выход")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice in {"0", "1", "2", "3"}:
            return int(choice)
        print("Ошибка: введите 0, 1, 2 или 3.")


def choose_mode() -> int:
    print("\nВыберите режим:")
    print("1 - Зашифровать")
    print("2 - Расшифровать")

    while True:
        choice = input("Ваш выбор: ").strip()
        if choice in {"1", "2"}:
            return int(choice)
        print("Ошибка: введите 1 или 2.")


def run_substitution(mode: int) -> None:
    text = input("\nВведите текст: ")
    key = input_substitution_key()

    try:
        if mode == 1:
            result = encrypt_substitution(text, key)
            print("\nРезультат шифрования:")
        else:
            result = decrypt_substitution(text, key)
            print("\nРезультат расшифрования:")

        print(result)
    except Exception as e:
        print(f"Ошибка: {e}")


def run_affine(mode: int) -> None:
    text = input("\nВведите текст: ")
    a, b = input_affine_key()

    try:
        if mode == 1:
            result = encrypt_affine(text, a, b)
            print("\nРезультат шифрования:")
        else:
            result = decrypt_affine(text, a, b)
            print("\nРезультат расшифрования:")

        print(result)
    except Exception as e:
        print(f"Ошибка: {e}")


def run_recurrent_affine(mode: int) -> None:
    text = input("\nВведите текст: ")
    a1, b1, a2, b2 = input_recurrent_affine_key()

    try:
        if mode == 1:
            result = encrypt_recurrent_affine(text, a1, b1, a2, b2)
            print("\nРезультат шифрования:")
        else:
            result = decrypt_recurrent_affine(text, a1, b1, a2, b2)
            print("\nРезультат расшифрования:")

        print(result)
    except Exception as e:
        print(f"Ошибка: {e}")


def print_info() -> None:
    print("=" * 60)
    print("Практическая работа 2. Подстановочные шифры")
    print("=" * 60)
    print(f"Используемый алфавит ({M} символа):")
    print(ALPHABET)
    print("\nПравило обработки символов вне ALPHABET:")
    print("Они не шифруются и остаются без изменений.")
    print("=" * 60)


def ask_continue() -> bool:
    while True:
        answer = input("\nВыполнить ещё одну операцию? (y/n): ").strip().lower()
        if answer in {"y", "yes", "д", "да"}:
            return True
        if answer in {"n", "no", "н", "нет"}:
            return False
        print("Ошибка: введите y/n.")


# =========================================================
# 8. main
# =========================================================

def main() -> None:
    print_info()

    while True:
        cipher_choice = choose_cipher()

        if cipher_choice == 0:
            print("\nВыход из программы.")
            break

        mode = choose_mode()

        if cipher_choice == 1:
            run_substitution(mode)
        elif cipher_choice == 2:
            run_affine(mode)
        elif cipher_choice == 3:
            run_recurrent_affine(mode)

        if not ask_continue():
            print("\nВыход из программы.")
            break


if __name__ == "__main__":
    main()