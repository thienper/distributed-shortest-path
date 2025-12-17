@echo off
REM Train GraphSAGE model for 50 epochs
cd /d "t:\NAM_4\NAM_4_HK1\DU LIEU LON\BT_LON\streaming-event-clustering\distributed-shortest-path"
call .venv\Scripts\activate.bat
python fast_train.py --graph large --samples 1000 --epochs 50
pause
