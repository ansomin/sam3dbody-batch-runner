@echo off
setlocal enabledelayedexpansion

SET ENV_NAME=sam_3d_body

REM Ensure conda is available in this shell
call conda.bat activate %ENV_NAME%
IF ERRORLEVEL 1 (
    echo Failed to activate conda env %ENV_NAME%
    exit /b 1
)

for /f "usebackq delims=" %%i in ("inputs_to_run.txt") do (
    echo ================================
    echo Running SAM-3D-Body on %%i
    echo ================================

    python run_batch_sam3d.py ^
      --input_dir "%%i" ^
      --device cuda ^
      --save_mesh ^
      --save_2d_vis

    IF ERRORLEVEL 1 (
        echo Failed on %%i
    ) ELSE (
        echo Completed %%i
    )
)

echo All jobs completed.
pause