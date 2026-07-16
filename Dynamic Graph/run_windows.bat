@echo off
REM Windows CMD
REM Default:
REM   run_windows.bat
REM Custom:
REM   run_windows.bat -- --dataset tgbl-wiki --max_events 10000 --epochs 10

python setup_and_run_tgb_dynarange.py %*
pause
