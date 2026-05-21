$ErrorActionPreference = 'Stop'

# AutoSegmentation 실행 파일을 Code\dist에 만든 뒤 프로젝트 루트로 복사하는 빌드 스크립트입니다.
$distDir = Join-Path $PSScriptRoot 'dist'
$workDir = Join-Path $PSScriptRoot 'build'
$targetExe = Join-Path (Split-Path $PSScriptRoot -Parent) 'AutoSegmentation.exe'
$builtExe = Join-Path $distDir 'AutoSegmentation.exe'

if (Test-Path -LiteralPath $distDir) {
    Remove-Item -LiteralPath $distDir -Recurse -Force
}

python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --windowed `
  --exclude-module qt_material `
  --add-data "training_runner.py;." `
  --add-data "auto_label_runner.py;." `
  --add-data "validation_runner.py;." `
  --name AutoSegmentation `
  --distpath $distDir `
  --workpath $workDir `
  --specpath . `
  main.py

# 기존 EXE를 먼저 삭제하지 않고 직접 덮어써 교체 실패 시 기존 실행 파일을 보존합니다.
$copySucceeded = $false
for ($attempt = 1; $attempt -le 10; $attempt++) {
    try {
        [System.IO.File]::Copy($builtExe, $targetExe, $true)
        $copySucceeded = $true
        break
    }
    catch {
        if ($attempt -eq 10) {
            throw
        }
        Start-Sleep -Milliseconds (200 * $attempt)
    }
}

# 최종 실행 파일만 남기기 위해 중간 dist 산출물은 빌드 후 정리합니다.
if ($copySucceeded -and (Test-Path -LiteralPath $distDir)) {
    Remove-Item -LiteralPath $distDir -Recurse -Force
}
