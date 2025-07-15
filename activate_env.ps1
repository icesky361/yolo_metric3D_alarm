<# Automatically load project .env file and activate virtual environment #>
# Complete encoding settings to resolve Chinese garbled characters
$OutputEncoding = [console]::InputEncoding = [console]::OutputEncoding = New-Object System.Text.UTF8Encoding
chcp 65001 | Out-Null

$envFile = Join-Path -Path $PWD.Path -ChildPath ".env"
if (Test-Path $envFile) {
    Write-Host "[Auto-activate] Loading environment file: $envFile"
    # Read .env file with UTF-8 encoding
    Get-Content -Path $envFile -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if (-not [string]::IsNullOrEmpty($line) -and -not $line.StartsWith('#')) {
            Write-Host "[Auto-activate] Executing command: $line"
            Invoke-Expression $line
        }
    }
} else {
    Write-Warning "[Auto-activate] Environment file not found: $envFile"
}