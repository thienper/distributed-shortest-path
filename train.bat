@echo off
REM Quick start training script for Windows

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║ GraphSAGE Model Training - Quick Start (Windows)      ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

setlocal enabledelayedexpansion

REM Set defaults
set GRAPH=medium
set EPOCHS=50

REM Parse arguments
if "%1%"=="" goto start_training
set GRAPH=%1%
if "%2%"=="" goto start_training
set EPOCHS=%2%

:start_training
REM Check if data files exist
if not exist "data\graphs" (
    echo ❌ Data directory not found: data\graphs
    echo Generating test graphs...
    python src\data_generation\graph_generator.py
)

REM Create model directory
if not exist "models" mkdir models
echo ✓ Model directory: models
echo.

echo 📊 Training Configuration:
echo   Graph size: %GRAPH%
echo   Epochs: %EPOCHS%
echo.

echo 🚀 Starting training...
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

python train_model.py --graph %GRAPH% --epochs %EPOCHS%

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo ✅ Training Complete!
echo.
echo 📁 Output files:
echo   • models\best_model.pt - Trained model weights
echo   • models\config.json - Training configuration
echo   • models\results.json - Metrics and results
echo.
echo 📊 Next step: Analyze results
echo   jupyter notebook notebooks\analyze_training_results.ipynb
echo.
echo 🚀 Next step: Use in predictor
echo   1. Update src\predictor\predictor_service.py
echo   2. Uncomment MLPredictor initialization
echo   3. Restart Flask: python -m web_app.app
echo.

pause
