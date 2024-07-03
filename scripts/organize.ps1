mkdir Ancillary
mkdir Spectra

mkdir Ancillary/A/
mkdir Ancillary/A/ccf/
mkdir Ancillary/A/e2ds/
mkdir Ancillary/A/s1d/
mkdir Ancillary/A/tables/
mkdir Ancillary/A/bis/

mkdir Ancillary/B/
mkdir Ancillary/B/ccf/
mkdir Ancillary/B/e2ds/
mkdir Ancillary/B/s1d/
mkdir Ancillary/B/tables/
mkdir Ancillary/B/bis/

mkdir Ancillary/intGuide

# Untar all HARPSTAR data into ./Ancillary/ and remove tar files after extraction
foreach ($file in (Get-ChildItem -Path ./archive/ -File -Recurse -Filter "*.tar"))
{tar xvf $file.FullName -C ./Ancillary/; rm $file.FullName}

# Organize spectra data
foreach ($file in (Get-ChildItem -Path ./archive/ -File -Filter "*.fits"))
{Move-Item -Path $file.FullName -Destination ./Spectra/}

# Organize A data
foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*ccf*A*.fits"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/A/ccf/}

foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*e2ds*A*.fits"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/A/e2ds/}

foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*s1d*A*.fits"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/A/s1d/}

foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*ccf*A*.tbl"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/A/tables/}

foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*bis*A*.fits"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/A/bis/}

# Organize B data
foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*ccf*B*.fits"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/B/ccf/}

foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*e2ds*B*.fits"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/B/e2ds/}

foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*s1d*B*.fits"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/B/s1d/}

foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*ccf*B*.tbl"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/B/tables/}

foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*bis*B*.fits"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/B/bis/}

# Organize INT_GUIDE data
foreach ($file in `
(Get-ChildItem -Path ./Ancillary/data/reduced/ -File -Recurse -Filter "*INT_GUIDE*.fits"))`
 {Move-Item -Path $file.FullName -Destination ./Ancillary/intGuide/}

# Rename ./archive/ folder
Rename-Item -Path ./archive/ -NewName ./README

# Remove data folder if there are no files left
if ((Get-ChildItem -Path ./Ancillary/data/ -Recurse -File | Measure-Object).Count -eq 0)
{Remove-Item -Recurse -Force ./Ancillary/data/} else {Write-Output "There are files left"}