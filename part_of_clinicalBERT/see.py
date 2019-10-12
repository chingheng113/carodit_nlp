import os
from pathlib import Path

current_path = os.path.dirname(__file__)
root_path = Path(current_path).parent

print(current_path)
print(root_path)