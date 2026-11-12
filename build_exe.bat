@echo off
rem Builds dist\QB Ranker.exe - a single program anyone on Windows can download and double-click.
rem First run sets up a small build environment (pandas, numpy, PyInstaller) in %LOCALAPPDATA%\QBRankerBuild.
cd /d "%~dp0"
set "BUILD=%LOCALAPPDATA%\QBRankerBuild"
set "PY=%BUILD%\venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Setting up the build environment...
    py -3.13 -m venv "%BUILD%\venv" || goto :fail
    "%PY%" -m pip install --upgrade pip || goto :fail
    "%PY%" -m pip install pandas numpy pyinstaller || goto :fail
)

if not exist "%~dp0model.json" (
    echo model.json is missing - train the prediction model first:  python predict.py --train
    goto :fail
)

echo Building QB Ranker.exe...
"%PY%" -m PyInstaller --noconfirm --clean --onefile --windowed --name "QB Ranker" ^
    --add-data "%~dp0weights.json;." ^
    --add-data "%~dp0model.json;." ^
    --distpath "%~dp0dist" --workpath "%BUILD%\work" --specpath "%BUILD%" ^
    --exclude-module pytest --exclude-module IPython --exclude-module matplotlib --exclude-module scipy ^
    "%~dp0app.py" || goto :fail

copy /y "%~dp0README.txt" "%~dp0dist\README.txt" >nul
echo.
echo Done: "%~dp0dist\QB Ranker.exe"
pause
exit /b 0

:fail
echo.
echo Build failed - see the messages above.
pause
exit /b 1
