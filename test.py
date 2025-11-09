import os
from dotenv import load_dotenv

# 🔧 Always work from project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")  # go one level up if this file is inside /src

print("Current Working Directory:", os.getcwd())

load_dotenv()
