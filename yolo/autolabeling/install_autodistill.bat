@echo off
echo Installing Autodistill and models...
echo.

echo Installing core Autodistill...
uv pip install autodistill

echo.
echo Installing Grounding DINO...
uv pip install autodistill-grounding-dino

echo.
echo Installing Grounded SAM (Grounding DINO + SAM)...
uv pip install autodistill-grounded-sam

echo.
echo Done! You can now use:
echo   - autodistill_autolabel.py (Grounding DINO only, bboxes)
echo   - autodistill_sam_autolabel.py (Grounding DINO + SAM, polygons)
echo.
pause
