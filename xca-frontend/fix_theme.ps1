$files = Get-ChildItem -Path "src" -Recurse -Filter "*.jsx"
foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw

    $content = $content -replace "fontFamily: \"'Inter', \-apple-system.*?\"", "fontFamily: ""'Source Sans 3', sans-serif"""
    $content = $content -replace "fontFamily: \"'Inter', sans-serif\"", "fontFamily: ""'Source Sans 3', sans-serif"""
    $content = $content -replace "fontFamily: \"'JetBrains Mono'.*?\"", "fontFamily: ""'IBM Plex Mono', monospace"""
    $content = $content -replace "fontFamily: 'Inter', sans-serif", "fontFamily: ""'Source Sans 3', sans-serif"""
    $content = $content -replace "fontFamily: 'JetBrains Mono', monospace", "fontFamily: ""'IBM Plex Mono', monospace"""

    Set-Content -Path $file.FullName -Value $content -NoNewline
}

$cssfile = "src/index.css"
if (Test-Path $cssfile) {
    $contentCss = Get-Content $cssfile -Raw
    $contentCss = $contentCss -replace "'Inter', Arial, sans-serif", "'Source Sans 3', Arial, sans-serif"
    $contentCss = $contentCss -replace "font-family: 'Inter', Arial, sans-serif", "font-family: 'Source Sans 3', Arial, sans-serif"
    Set-Content -Path $cssfile -Value $contentCss -NoNewline
}

Write-Host "Fonts updated to Audit Paper."
