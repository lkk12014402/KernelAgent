import os

DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"
NUM_OF_ROUNDS = int(os.getenv("NUM_OF_ROUNDS", "20"))
TIMEOUT = int(os.getenv("TIMEOUT", "2"))
MAX_TOKENS = 4096
