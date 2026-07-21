@echo off
echo =========================================================
echo       Vision Tools Local Web Applications Launcher
echo =========================================================
echo.
echo Starting Edge Detection Web App...
echo Address: http://localhost:5001
start "Edge Detection Server" py EdgeDetection/app.py
echo.
echo Starting Face Detection Web App...
echo Address: http://localhost:5002
start "Face Detection Server" py facedetection/app.py
echo.
echo =========================================================
echo Both servers are launching in separate windows!
echo - Edge Detection: http://localhost:5001
echo - Face Detection: http://localhost:5002
echo =========================================================
echo.
pause
