import subprocess

with open('full_error.txt', 'w') as f:
    subprocess.run(['python', 'debug_db.py'], stdout=f, stderr=f)
