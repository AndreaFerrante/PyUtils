import secrets
import string

_ALPHABET = string.ascii_letters + string.digits + string.punctuation


def generate_password(length: int = 12) -> str:
    """Generate a cryptographically secure random password of the given length."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))
