@echo off
chcp 65001 >nul 2>&1
title JB-Computer Chatbot Monitor
cd /d "%~dp0"

echo ============================================
echo   JB-Computer Chatbot Monitor - Startup
echo ============================================
echo.

:: -----------------------------------------------
:: 1. Admin-Rechte holen (fuer Firewall + Netzwerk)
:: -----------------------------------------------
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Starte mit Admin-Rechten...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: -----------------------------------------------
:: 2. Netzwerk + Firewall
:: -----------------------------------------------
powershell -Command "Set-NetConnectionProfile -InterfaceAlias 'Ethernet' -NetworkCategory Private" >nul 2>&1
echo [OK] Netzwerkprofil: Privat

netsh advfirewall firewall delete rule name="Chatbot Monitor 7779" >nul 2>&1
netsh advfirewall firewall add rule name="Chatbot Monitor 7779" dir=in action=allow protocol=TCP localport=7779 >nul 2>&1
echo [OK] Firewall: Port 7779 offen

echo.
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4"') do echo [INFO] LAN-Zugriff: http://%%a:7779
echo [INFO] Lokal:       http://localhost:7779
echo.

:: -----------------------------------------------
:: 3. Ollama starten (falls nicht schon laeuft)
:: -----------------------------------------------
echo [....] Pruefe Ollama...
curl -s http://localhost:11434/api/version >nul 2>&1
if %errorLevel% neq 0 (
    echo [....] Ollama wird gestartet...
    start "" "C:\Users\User\AppData\Local\Programs\Ollama\ollama.exe" serve
    :: Warten bis Ollama bereit ist (max 30 Sek)
    for /L %%i in (1,1,30) do (
        timeout /t 1 /nobreak >nul
        curl -s http://localhost:11434/api/version >nul 2>&1 && goto ollama_ready
    )
    echo [WARN] Ollama antwortet nicht - Embeddings werden nicht funktionieren!
    goto ollama_done
)
:ollama_ready
echo [OK] Ollama laeuft
:ollama_done

:: -----------------------------------------------
:: 4. Embedding-Modell sicherstellen
:: -----------------------------------------------
echo [....] Pruefe Embedding-Modell (nomic-embed-text)...
curl -s http://localhost:11434/api/tags 2>nul | findstr /i "nomic-embed-text" >nul 2>&1
if %errorLevel% neq 0 (
    echo [....] Lade nomic-embed-text herunter (einmalig, ~274 MB)...
    "C:\Users\User\AppData\Local\Programs\Ollama\ollama.exe" pull nomic-embed-text
    if %errorLevel% equ 0 (
        echo [OK] nomic-embed-text installiert
    ) else (
        echo [WARN] Download fehlgeschlagen - RAG-Embeddings funktionieren nicht
    )
) else (
    echo [OK] nomic-embed-text vorhanden
)

:: -----------------------------------------------
:: 5. Python-Pakete pruefen
:: -----------------------------------------------
echo [....] Pruefe Python-Pakete...
python -c "import openpyxl" >nul 2>&1 || (
    echo [....] Installiere openpyxl...
    pip install openpyxl -q
)
python -c "import PyPDF2" >nul 2>&1 || (
    echo [....] Installiere PyPDF2...
    pip install PyPDF2 -q
)
python -c "import numpy" >nul 2>&1 || (
    echo [....] Installiere numpy...
    pip install numpy -q
)
python -c "import tiktoken" >nul 2>&1 || (
    echo [....] Installiere tiktoken...
    pip install tiktoken -q
)
echo [OK] Python-Pakete bereit

:: -----------------------------------------------
:: 6. Knowledge-Ordner erstellen (falls noetig)
:: -----------------------------------------------
if not exist "knowledge" mkdir knowledge
echo [OK] Knowledge-Ordner: %cd%\knowledge

:: -----------------------------------------------
:: 7. Alten Server stoppen (falls noch laeuft)
:: -----------------------------------------------
echo [....] Pruefe ob alter Server laeuft...
curl -s http://localhost:7779/api/health >nul 2>&1
if %errorLevel% equ 0 (
    echo [....] Alter Server wird gestoppt...
    powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -like '*Chatbot*' -or $_.CommandLine -like '*server.py*'} | Stop-Process -Force" >nul 2>&1
    timeout /t 2 /nobreak >nul
)

:: -----------------------------------------------
:: 8. Server starten + Browser oeffnen
:: -----------------------------------------------
echo.
echo ============================================
echo   Server startet auf Port 7779...
echo   Dieses Fenster NICHT schliessen!
echo ============================================
echo.

start "" http://localhost:7779
python server.py
pause
