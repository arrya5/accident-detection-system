@echo off
title Accident Detection System - Running...
echo.
echo  Starting Accident Detection...
echo  Please wait for the video window to open.
echo.
"c:\documents\COLLEGE\Minor Project (Accident Detection)\.venv\Scripts\python.exe" "c:\documents\COLLEGE\Minor Project (Accident Detection)\src\detect_pytorch.py" --source "c:\documents\COLLEGE\Minor Project (Accident Detection)\test_video.mp4" --output "c:\documents\COLLEGE\Minor Project (Accident Detection)\output\test_video_result.mp4" --threshold 0.6
echo.
echo  Detection complete! Press any key to close.
pause > nul
