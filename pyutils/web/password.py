import random, string

def generate_password(length:str=12) -> str :

    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))