@echo off
setlocal enabledelayedexpansion
CALL "C:\ProgramData\anaconda3\Scripts\activate.bat"

:dependency_check
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
:run_ai
python ai.py

:set_exit_option
echo.
set /p UserExit=Do you want to exit? [Y/N]: 
IF /I "!UserExit!"=="Y" (
    goto end_script
) ELSE (
    echo Restarting ai.py...
    goto run_ai
)

:end_script
pause
endlocal