import os, sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join('/data/linc9/carodit_nlp/')))

current_path = os.getcwd()
root_path = Path(current_path).parent

print(current_path)
print(root_path)