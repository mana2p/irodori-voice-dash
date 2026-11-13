# PowerShell script to download ONNX models for Irodori-TTS
param(
    [string]$ModelsDir
)
$ErrorActionPreference = "Stop"

$env:NO_PROXY = "localhost,127.0.0.1"
$env:no_proxy = "localhost,127.0.0.1"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseDir = Split-Path -Parent $ScriptDir
if (-not $ModelsDir) {
    $ModelsDir = Join-Path $BaseDir "models"
}

function Download-File {
    param(
        [string]$Url,
        [string]$Dest
    )
    $destDir = Split-Path -Parent $Dest
    if (-not (Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    if ((Test-Path $Dest) -and ((Get-Item $Dest).Length -gt 0)) {
        Write-Host "Skip existing: $Dest"
        return
    }
    Write-Host "Downloading: $Url -> $Dest"
    # Use curl.exe if available for robust resume & progress, else Invoke-WebRequest
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe --ssl-no-revoke -L -C - --retry 3 --retry-delay 5 -o $Dest $Url
    }
    else {
        Invoke-WebRequest -Uri $Url -OutFile $Dest
    }
}

$B = "https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main"
Download-File "$B/text_encoder_int8.onnx" (Join-Path $ModelsDir "int8\text_encoder_int8.onnx")
Download-File "$B/dit_step_int8.onnx" (Join-Path $ModelsDir "int8\dit_step_int8.onnx")
Download-File "$B/duration.onnx" (Join-Path $ModelsDir "int8\duration.onnx")
Download-File "$B/context_kv.onnx" (Join-Path $ModelsDir "int8\context_kv.onnx")
Download-File "$B/speaker_encoder.onnx" (Join-Path $ModelsDir "int8\speaker_encoder.onnx")
Download-File "$B/dacvae_decode.onnx" (Join-Path $ModelsDir "int8\dacvae_decode.onnx")
Download-File "$B/config.json" (Join-Path $ModelsDir "int8\config.json")
Download-File "$B/manifest.json" (Join-Path $ModelsDir "int8\manifest.json")
Download-File "$B/tokenizer/tokenizer.json" (Join-Path $ModelsDir "int8\tokenizer\tokenizer.json")
Download-File "$B/tokenizer/tokenizer_config.json" (Join-Path $ModelsDir "int8\tokenizer\tokenizer_config.json")

$F = "https://huggingface.co/raratu/Onsei-iOS-Models/resolve/main/v4.1-small-mf"
Download-File "$F/text_encoder.onnx" (Join-Path $ModelsDir "f32\text_encoder.onnx")
Download-File "$F/dit_step.onnx" (Join-Path $ModelsDir "f32\dit_step.onnx")
Download-File "$F/dacvae_encode.onnx" (Join-Path $ModelsDir "f32\dacvae_encode.onnx")
Download-File "$F/config.json" (Join-Path $ModelsDir "f32\config.json")
Download-File "$F/tokenizer/tokenizer.json" (Join-Path $ModelsDir "f32\tokenizer\tokenizer.json")
Download-File "$F/tokenizer/tokenizer_config.json" (Join-Path $ModelsDir "f32\tokenizer\tokenizer_config.json")

Write-Host "DONE! All models downloaded to $ModelsDir"

