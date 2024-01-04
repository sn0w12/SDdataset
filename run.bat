@echo off
setlocal enabledelayedexpansion
CALL "C:\ProgramData\anaconda3\Scripts\activate.bat"

python check_dependencies.py > missing_packages.txt
SET DEPENDENCY_CHECK=%ERRORLEVEL%
IF !DEPENDENCY_CHECK! EQU 1 (
    echo Missing packages found.
    set /p UserResponse=Do you want to install missing packages? [Y/N]: 
    IF /I "!UserResponse!"=="Y" (
        echo Installing missing packages...
        for /F "tokens=*" %%i in (missing_packages.txt) do pip install %%i
    ) ELSE (
        echo Skipping package installation.
    )
)

echo Running ai.py...
python ai.py
pause
endlocal