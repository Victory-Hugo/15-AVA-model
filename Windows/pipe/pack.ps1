#!/usr/bin/env pwsh
# -*- coding: utf-8 -*-

<#
.SYNOPSIS
    自动打包脚本 - 将 1-pipe.py 打包为 AutoAdmix_v1.exe
    
.DESCRIPTION
    使用 PyInstaller 将 1-pipe.py 打包为可执行文件 AutoAdmix_v1.exe
    确保 conda base 环境中安装了 PyInstaller
    
.EXAMPLE
    ./pack.ps1
#>

# 设置错误处理
$ErrorActionPreference = "Stop"

# 获取当前脚本所在目录
$scriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$pipeDir = $scriptDir  # 当前目录就是 pipe 目录

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "AVA Model Pipeline - 自动打包脚本" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# 检查必要文件
Write-Host "[1/4] 检查必要文件..." -ForegroundColor Yellow
$requiredFiles = @("1-pipe.py", "AVA-Pipeline.spec")
foreach ($file in $requiredFiles) {
    $filePath = Join-Path $pipeDir $file
    if (-not (Test-Path $filePath)) {
        Write-Host "❌ 错误: 找不到文件 $file" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ 找到 $file" -ForegroundColor Green
}

# 检查 conda 环境
Write-Host ""
Write-Host "[2/4] 检查 Conda 环境..." -ForegroundColor Yellow
try {
    $condaVersion = conda --version 2>&1
    Write-Host "  ✓ $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ 错误: 找不到 conda" -ForegroundColor Red
    exit 1
}

# 检查 PyInstaller
Write-Host ""
Write-Host "[3/4] 检查 PyInstaller..." -ForegroundColor Yellow
try {
    $pyinstaller = conda list | findstr /I "pyinstaller"
    if ($pyinstaller) {
        Write-Host "  ✓ PyInstaller 已安装: $($pyinstaller -split ' ' | Select-Object -Index 1)" -ForegroundColor Green
    } else {
        Write-Host "⚠ 警告: 未找到 PyInstaller，尝试从 pip 调用" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ 警告: 无法检查 PyInstaller" -ForegroundColor Yellow
}

# 清理旧的编译结果
Write-Host ""
Write-Host "[4/4] 清理旧的编译结果..." -ForegroundColor Yellow
if (Test-Path (Join-Path $pipeDir "build")) {
    Remove-Item -Path (Join-Path $pipeDir "build") -Recurse -Force
    Write-Host "  ✓ 删除 build 目录" -ForegroundColor Green
}

# 保留 dist 目录但清空其中的文件
if (-not (Test-Path (Join-Path $pipeDir "dist"))) {
    New-Item -Path (Join-Path $pipeDir "dist") -ItemType Directory | Out-Null
    Write-Host "  ✓ 创建 dist 目录" -ForegroundColor Green
} else {
    Write-Host "  ✓ dist 目录已存在" -ForegroundColor Green
}

# 执行打包
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "开始打包..." -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

try {
    # 激活 conda base 环境并运行 PyInstaller
    conda activate base
    python -m PyInstaller --distpath dist --specpath . AVA-Pipeline.spec
    
    # 检查打包结果
    $exePath = Join-Path $pipeDir "dist\AutoAdmix_v1.exe"
    if (Test-Path $exePath) {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
        Write-Host "✓ 打包成功！" -ForegroundColor Green
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
        Write-Host ""
        Write-Host "输出文件: $exePath" -ForegroundColor Cyan
        
        # 获取文件大小
        $fileSize = (Get-Item $exePath).Length / 1MB
        Write-Host "文件大小: {0:F2} MB" -ForegroundColor Cyan -f $fileSize
        
        Write-Host ""
        Write-Host "您可以运行以下命令启动应用:" -ForegroundColor Yellow
        Write-Host "  .$exePath" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "❌ 打包失败: 找不到生成的 exe 文件" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host ""
    Write-Host "❌ 打包过程中出错: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
