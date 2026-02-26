rem pip install -r requirements.txt --upgrade
py -m nuitka v3mf.py --mode=onefile --include-package=SourceIO --follow-imports --onefile-no-dll --enable-plugin=pyqt6 --enable-plugin=dill-compat