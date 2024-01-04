@echo off
CALL "C:\ProgramData\anaconda3\Scripts\activate.bat"
python ai.py
IF %ERRORLEVEL% EQU 1 (
    echo Installing missing packages...
    pip install -r requirements.txt
	cls
	python ai.py
)
pause