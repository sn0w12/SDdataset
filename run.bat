@echo off
CALL "C:\ProgramData\anaconda3\Scripts\activate.bat"
IF %ERRORLEVEL% EQU 1 (
    echo Installing missing packages...
    pip install -r requirements.txt
)
python ai.py
pause