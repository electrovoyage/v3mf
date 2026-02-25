pip3 install -r requirements.txt --break-system-packages
python3 -m nuitka v3mf.py --mode=onefile --include-package=SourceIO --follow-imports --onefile-no-dll