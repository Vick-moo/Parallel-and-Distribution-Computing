# ---------------------------------------------------------------------------
#  Renders an Excel-style summary chart:
#    Sequential Time Ts (ms) vs. Matrix Size (m x n)
#
#  Usage (defaults keep the original sequential figure):
#      powershell -ExecutionPolicy Bypass -File make_chart.ps1
#
#  Override for another run:
#      powershell -ExecutionPolicy Bypass -File make_chart.ps1 `
#          -YMax 7200 -Title 'My title' -Out my.png `
#          -Ms @(1,2,3,4,5)
# ---------------------------------------------------------------------------

param(
    [string] $Title = 'Sequential Time Ts (ms) vs. Matrix Size (m x n)',
    [double] $YMax  = 12000,
    [double] $YStep = 2000,
    [string] $Ms    = '213.054,769.972,1903.127,3856.368,6700.946,10493.166',
    [string] $Out   = 'sequential_time_vs_size.png'
)

$msValues = @($Ms -split ',' | ForEach-Object { [double]$_.Trim() })

Add-Type -AssemblyName System.Windows.Forms.DataVisualization

$sizes = @('1000 x 1000', '1500 x 1500', '2000 x 2000', '2500 x 2500', '3000 x 3000', '3500 x 3500')
$mem   = @('~8 Mb',      '~18 Mb',      '~32 Mb',      '~50 Mb',      '~72 Mb',      '~98 Mb')

$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$chart.Width   = 1156
$chart.Height  = 706
$chart.BackColor = [System.Drawing.Color]::White
$chart.AntiAliasing = [System.Windows.Forms.DataVisualization.Charting.AntiAliasingStyles]::All

# ---- title (left aligned, light grey, like Excel) --------------------------
$chartTitle = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$chartTitle.Text      = $Title
$chartTitle.Font      = New-Object System.Drawing.Font('Segoe UI', 15, [System.Drawing.FontStyle]::Regular)
$chartTitle.ForeColor = [System.Drawing.ColorTranslator]::FromHtml('#A6A6A6')
$chartTitle.Alignment = [System.Drawing.ContentAlignment]::MiddleLeft
$chartTitle.Docking   = [System.Windows.Forms.DataVisualization.Charting.Docking]::Top
$chartTitle.Position  = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(1.0, 1.5, 95, 8)
$chart.Titles.Add($chartTitle)

# ---- single chart area ----------------------------------------------------
$ca = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea
$ca.Name = 'Main'
$ca.BackColor = [System.Drawing.Color]::White
$ca.Position  = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(9.5, 17, 89, 77)
$chart.ChartAreas.Add($ca)

$axisFont   = New-Object System.Drawing.Font('Segoe UI', 9)
$labelFont  = New-Object System.Drawing.Font('Segoe UI', 8.2)

# X axis: categorical, two-line labels
$ca.AxisX.MajorGrid.Enabled      = $false
$ca.AxisX.LineColor              = [System.Drawing.ColorTranslator]::FromHtml('#BFBFBF')
$ca.AxisX.LineWidth              = 1
$ca.AxisX.MajorTickMark.Enabled  = $false
$ca.AxisX.LabelStyle.Font        = $labelFont
$ca.AxisX.LabelStyle.ForeColor   = [System.Drawing.ColorTranslator]::FromHtml('#595959')
$ca.AxisX.IsMarginVisible        = $true
$ca.AxisX.Title                  = 'Matrix Size (m x n)'
$ca.AxisX.TitleFont              = $axisFont
$ca.AxisX.TitleForeColor         = [System.Drawing.ColorTranslator]::FromHtml('#595959')

# Y axis: gridlines, 3 decimals
$ca.AxisY.Minimum                = 0
$ca.AxisY.Maximum                = $YMax
$ca.AxisY.Interval               = $YStep
$ca.AxisY.LabelStyle.Format      = '0.000'
$ca.AxisY.LabelStyle.Font        = $labelFont
$ca.AxisY.LabelStyle.ForeColor   = [System.Drawing.ColorTranslator]::FromHtml('#595959')
$ca.AxisY.MajorGrid.LineColor    = [System.Drawing.ColorTranslator]::FromHtml('#E7E6E6')
$ca.AxisY.MajorGrid.LineWidth    = 1
$ca.AxisY.MajorTickMark.Enabled  = $false
$ca.AxisY.LineColor              = [System.Drawing.ColorTranslator]::FromHtml('#BFBFBF')
$ca.AxisY.LineWidth              = 1
$ca.AxisY.Title                  = 'Sequential Time Ts (ms)'
$ca.AxisY.TitleFont              = $axisFont
$ca.AxisY.TitleForeColor         = [System.Drawing.ColorTranslator]::FromHtml('#595959')

# ---- legend under the title ------------------------------------------------
$legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
$legend.Name            = 'Legend1'
$legend.Docking         = [System.Windows.Forms.DataVisualization.Charting.Docking]::Top
$legend.Alignment       = [System.Drawing.StringAlignment]::Center
$legend.Font            = $axisFont
$legend.ForeColor       = [System.Drawing.ColorTranslator]::FromHtml('#595959')
$legend.BackColor       = [System.Drawing.Color]::White
$legend.BorderColor     = [System.Drawing.Color]::Transparent
$legend.IsTextAutoFit   = $false
$legend.Position        = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(0, 8, 100, 5)
$chart.Legends.Add($legend)

# ---- series 1: the bars ---------------------------------------------------
$bars = New-Object System.Windows.Forms.DataVisualization.Charting.Series
$bars.Name        = 'bars'
$bars.LegendText  = 'Measured Ts (bar)'
$bars.ChartArea   = 'Main'
$bars.ChartType   = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Column
$bars.Legend      = 'Legend1'
$bars.Color       = [System.Drawing.ColorTranslator]::FromHtml('#4472C4')
$bars.BorderColor = [System.Drawing.ColorTranslator]::FromHtml('#4472C4')
$bars['PointWidth'] = '0.62'
$chart.Series.Add($bars)

# ---- series 2: the red trend line -----------------------------------------
$line = New-Object System.Windows.Forms.DataVisualization.Charting.Series
$line.Name        = 'trend'
$line.LegendText  = 'Ts trend (line)'
$line.ChartArea   = 'Main'
$line.ChartType   = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Line
$line.Legend      = 'Legend1'
$line.Color       = [System.Drawing.ColorTranslator]::FromHtml('#FF0000')
$line.BorderWidth = 2
$line.BorderDashStyle = [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Solid
$line.MarkerStyle = [System.Windows.Forms.DataVisualization.Charting.MarkerStyle]::Diamond
$line.MarkerSize  = 8
$line.MarkerColor = [System.Drawing.ColorTranslator]::FromHtml('#FF0000')
$line['PointWidth'] = '0.62'
$chart.Series.Add($line)

for ($i = 0; $i -lt $sizes.Count; $i++) {
    $label = "$($sizes[$i])`n($($mem[$i]))"
    $null = $bars.Points.AddXY($label, $msValues[$i])
    $null = $line.Points.AddXY($label, $msValues[$i])
}

# ---- save -----------------------------------------------------------------
$out = Join-Path $PSScriptRoot $Out
$chart.SaveImage($out, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
$chart.Dispose()

Write-Host "saved: $out"
