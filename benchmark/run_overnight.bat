@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0.."

set "PYTHON_EXE=C:\ProgramData\anaconda3\envs\grcwa_mwa\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: configured grcwa interpreter not found:
    echo        %PYTHON_EXE%
    exit /b 2
)

set "PROFILE=night"
set "GRCWA_VARIANTS=fork"
rem Keep dense low 1D orders and matching total-order anchors q^2.
set "GRCWA_NG1D="
set "GRCWA_NG1D_FROM_Q2D=1"
rem Plot once at q<=15, then after every newly appended odd q through 61.
set "FULL_Q_LIST=1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61"
set "TARGET_Q_LIST=%FULL_Q_LIST%"
set "INITIAL_Q_LIST=1,3,5,7,9,11,13,15"
set "INITIAL_Q_MAX=15"
set "GROW_ORDERS=17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61"
set "STAGE_TOTAL=24"
set "GRCWA_MAX2D=3721"
set "GRCWA_FAST_REPEAT=3"
set "GRCWA_FAST_THRESHOLD_MS=1000"
set "GRCWA_CONV_TOL=1e-4"
set "GRCWA_CACHE=1"
set "GRCWA_CACHE_DIR=benchmark\.cache\convergence"
set "GRCWA_REFRESH_TIMING=0"
set "GRCWA_REQUIRED_COLUMNS=fork[Laurent],fork[Pol],ikarus[Laurent],ikarus[Li],ikarus[NV]"
set "GRCWA_OUTPUT_DIR=%CD%\benchmark"
set "GRCWA_PLOT_OUTPUT_DIR=%CD%\benchmark"
set "GRCWA_CONV_JSON=%CD%\benchmark\conv_results.json"
set "GRCWA_MOOSE_JSON=%CD%\benchmark\moose_reference.json"

if not "%~2"=="" goto :usage
if "%~1"=="" goto :profile_ready
if /I "%~1"=="quick" (
    set "PROFILE=quick"
    set "INITIAL_Q_LIST=1,3"
    set "INITIAL_Q_MAX=3"
    set "GROW_ORDERS=5"
    set "TARGET_Q_LIST=1,3,5"
    set "STAGE_TOTAL=2"
    set "GRCWA_MAX2D=25"
    goto :profile_ready
)
if /I "%~1"=="refresh-timing" (
    set "PROFILE=night_refresh_timing"
    set "INITIAL_Q_LIST=!FULL_Q_LIST!"
    set "INITIAL_Q_MAX=61"
    set "GROW_ORDERS="
    set "STAGE_TOTAL=1"
    set "GRCWA_REFRESH_TIMING=1"
    goto :profile_ready
)
:usage
echo Usage: %~nx0 [quick^|refresh-timing]
exit /b 2

:profile_ready
set "GRCWA_Q2D=%INITIAL_Q_LIST%"
set "LOG=benchmark\benchmark_%PROFILE%.log"
> "%LOG%" (
    echo Benchmark profile: %PROFILE%
    echo Started: %DATE% %TIME%
    echo Interpreter: %PYTHON_EXE%
    echo GRCWA_VARIANTS=%GRCWA_VARIANTS%
    echo GRCWA_NG1D_FROM_Q2D=%GRCWA_NG1D_FROM_Q2D% ^(1D uses sorted q union q^^2^)
    echo Initial GRCWA_Q2D=%GRCWA_Q2D%
    echo Final q schedule=%TARGET_Q_LIST%
    echo Snapshot stages=%STAGE_TOTAL%
    echo GRCWA_MAX2D=%GRCWA_MAX2D%
    echo GRCWA_FAST_REPEAT=%GRCWA_FAST_REPEAT%
    echo GRCWA_FAST_THRESHOLD_MS=%GRCWA_FAST_THRESHOLD_MS%
    echo GRCWA_CONV_TOL=%GRCWA_CONV_TOL%
    echo GRCWA_CACHE=%GRCWA_CACHE%
    echo GRCWA_CACHE_DIR=%GRCWA_CACHE_DIR%
    echo GRCWA_REFRESH_TIMING=%GRCWA_REFRESH_TIMING%
    echo GRCWA_REQUIRED_COLUMNS=%GRCWA_REQUIRED_COLUMNS%
    echo GRCWA_OUTPUT_DIR=%GRCWA_OUTPUT_DIR%
    echo GRCWA_PLOT_OUTPUT_DIR=%GRCWA_PLOT_OUTPUT_DIR%
    echo GRCWA_CONV_JSON=%GRCWA_CONV_JSON%
)
"%PYTHON_EXE%" --version >> "%LOG%" 2>&1

set /a "STAGE_INDEX=1"
call :run_snapshot "%INITIAL_Q_MAX%"
if errorlevel 1 goto :failed

if not defined GROW_ORDERS goto :growth_complete
for %%Q in (%GROW_ORDERS%) do (
    set "GRCWA_Q2D=!GRCWA_Q2D!,%%Q"
    set /a "STAGE_INDEX+=1"
    call :run_snapshot "%%Q"
    if errorlevel 1 goto :failed
)

:growth_complete
call :run_stage "Moose comparison plots" "benchmark\plot_moose.py"
if errorlevel 1 goto :failed

echo Completed: %DATE% %TIME%>> "%LOG%"
echo.
echo Benchmark completed successfully. Log: %CD%\%LOG%
exit /b 0

:run_snapshot
set "CURRENT_Q_MAX=%~1"
set /a "CURRENT_NG2D=CURRENT_Q_MAX*CURRENT_Q_MAX"
echo.
echo [stage !STAGE_INDEX!/!STAGE_TOTAL!] Expanding through q=!CURRENT_Q_MAX! ^(2D nG=!CURRENT_NG2D!^)...
>> "%LOG%" echo.
>> "%LOG%" echo ----- snapshot !STAGE_INDEX!/!STAGE_TOTAL!: q_max=!CURRENT_Q_MAX! -----
>> "%LOG%" echo GRCWA_NG1D=sorted union of q and q^^2
>> "%LOG%" echo GRCWA_Q2D=!GRCWA_Q2D!
call :run_stage "convergence sweep through q=!CURRENT_Q_MAX!" "benchmark\conv_run.py"
if errorlevel 1 exit /b 1
call :run_stage "convergence plots through q=!CURRENT_Q_MAX!" "benchmark\plot_conv.py"
if errorlevel 1 exit /b 1
exit /b 0

:run_stage
echo [%TIME%] Running %~1...
echo.>> "%LOG%"
echo ===== %~1 =====>> "%LOG%"
set "BENCHMARK_SCRIPT=%~2"
powershell.exe -NoLogo -NoProfile -Command ^
    "$utf8 = New-Object System.Text.UTF8Encoding($false); $writer = New-Object System.IO.StreamWriter($env:LOG, $true, $utf8); try { & $env:PYTHON_EXE -u $env:BENCHMARK_SCRIPT 2>&1 | ForEach-Object { $line = $_.ToString(); [Console]::Out.WriteLine($line); $writer.WriteLine($line); $writer.Flush() }; $rc = $LASTEXITCODE } finally { $writer.Dispose() }; exit $rc"
if errorlevel 1 (
    echo ERROR: %~1 failed. See %CD%\%LOG%
    exit /b 1
)
exit /b 0

:failed
echo Benchmark stopped after a failed stage. See %CD%\%LOG%
exit /b 1
