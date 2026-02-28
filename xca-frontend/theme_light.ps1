$files = Get-ChildItem -Path "src" -Recurse -Filter "*.jsx"
foreach ($file in $files) {
    if ($file.Name -match "AnalyticsDashboard") { continue }
    $content = Get-Content $file.FullName -Raw
    
    # Fonts
    $content = $content -replace "'Inter', (?:-apple-system, BlinkMacSystemFont, 'Segoe UI', )?sans-serif", "'Source Sans 3', sans-serif"
    $content = $content -replace "'Inter', sans-serif", "'Source Sans 3', sans-serif"
    $content = $content -replace "'JetBrains Mono', monospace", "'IBM Plex Mono', monospace"

    # Backgrounds & borders
    $content = $content.Replace("#09090b", "#f8f9fc")
    $content = $content.Replace("#18181b", "#ffffff")
    $content = $content.Replace("#27272a", "#d1d9e6")
    $content = $content.Replace("#1a1a1a", "#d1d9e6")
    $content = $content.Replace("#1a1a1f", "#f1f4f9")
    $content = $content.Replace("#1f1f23", "#f8f9fc")
    $content = $content.Replace("#0f0f11", "#f8f9fc")
    $content = $content.Replace("#0a0a0a", "#f1f4f9")
    
    # Text
    $content = $content.Replace("#fafafa", "#0f172a")
    $content = $content.Replace("#a1a1aa", "#475569")
    $content = $content.Replace("#52525b", "#475569")
    $content = $content.Replace("#71717a", "#64748b")
    $content = $content.Replace("#3f3f46", "#94a3b8")
    
    # Accents
    $content = $content.Replace("#3b82f6", "#1e40af")
    $content = $content.Replace("59,130,246", "30,64,175")
    $content = $content.Replace("#10b981", "#16a34a")
    $content = $content.Replace("16,185,129", "22,163,74")
    $content = $content.Replace("#f59e0b", "#d97706")
    $content = $content.Replace("245,158,11", "217,119,6")
    $content = $content.Replace("#ef4444", "#dc2626")
    $content = $content.Replace("239,68,68", "220,38,38")
    $content = $content.Replace("#f43f5e", "#be123c")
    $content = $content.Replace("244,63,94", "190,18,60")
    $content = $content.Replace("#8b5cf6", "#4f46e5")

    # Remove text-shadows using proper escape
    $content = $content -replace 'textShadow:\s*''.*?''', "textShadow: 'none'"
    $content = $content -replace 'textShadow:\s*".*?"', "textShadow: 'none'"
    $regexBacktick = 'textShadow:\s*`.*?`'
    $content = $content -replace $regexBacktick, "textShadow: 'none'"

    Set-Content -Path $file.FullName -Value $content -NoNewline
}

$jsFiles = Get-ChildItem -Path "src\hooks" -Recurse -Filter "*.js"
foreach ($file in $jsFiles) {
    $content = Get-Content $file.FullName -Raw
    $content = $content.Replace("#ffffff0a", "#0f172a08")
    $content = $content.Replace("#e4e4e7", "#0f172a")
    $content = $content.Replace("#52525b88", "#47556988")
    $content = $content.Replace("Courier New", "IBM Plex Mono")
    Set-Content -Path $file.FullName -Value $content -NoNewline
}
Write-Host "Color update complete."
