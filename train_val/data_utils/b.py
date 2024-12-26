from pathlib import Path


# Other Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO

print(FILE)
print(ROOT)