@echo off
chcp 65001 >nul
title 号彩智算 - 彩票分析预测平台
cd /d "%~dp0"
echo 正在启动号彩智算平台...
python app.py
if errorlevel 1 (
  echo.
  echo 启动失败：请确认已安装 Python 3.8+ 并加入 PATH。
  pause
)
