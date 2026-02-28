$files = Get-ChildItem -Path "src\components" -Recurse -Filter "*.jsx"
foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    # Bump small font sizes up
    $content = $content.Replace('fontSize: 8,', 'fontSize: 11,')
    $content = $content.Replace('fontSize: 9,', 'fontSize: 11,')
    $content = $content.Replace('fontSize: 9 ', 'fontSize: 11 ')
    $content = $content.Replace('fontSize: 10,', 'fontSize: 12,')
    $content = $content.Replace('fontSize: 10 ', 'fontSize: 12 ')
    $content = $content.Replace('fontSize: 10}', 'fontSize: 12}')
    $content = $content.Replace('fontSize: 11,', 'fontSize: 13,')
    $content = $content.Replace('fontSize: 11 ', 'fontSize: 13 ')
    $content = $content.Replace('fontSize: 11}', 'fontSize: 13}')
    $content = $content.Replace('fontSize: 12,', 'fontSize: 14,')
    $content = $content.Replace('fontSize: 12 ', 'fontSize: 14 ')
    $content = $content.Replace('fontSize: 12}', 'fontSize: 14}')
    Set-Content -Path $file.FullName -Value $content -NoNewline
}
Write-Host "Done. Bumped font sizes in $($files.Count) files."
