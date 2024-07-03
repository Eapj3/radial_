#! /bin/bash
# Run in same directory as 'archive' directory

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

mkdir Ancillary/intGuide/

# Untar all HARPSTAR data into ./Ancillary/ and remove tar files after extraction
find ./archive/ -iname '*.tar' -exec tar xvf '{}' -C ./Ancillary/ \; -exec rm '{}' \;

#Organize A data
find ./Ancillary/data/reduced/ -type f -iname '*ccf*A*.fits' -exec mv '{}' ./Ancillary/A/ccf/ \;

find ./Ancillary/data/reduced/ -type f -iname '*e2ds*A*.fits' -exec mv '{}' ./Ancillary/A/e2ds/ \;

find ./Ancillary/data/reduced/ -type f -iname '*s1d*A*.fits' -exec mv '{}' ./Ancillary/A/s1d/ \;

find ./Ancillary/data/reduced/ -type f -iname '*ccf*A*.tbl' -exec mv '{}' ./Ancillary/A/tables/ \;

find ./Ancillary/data/reduced/ -type f -iname '*bis*A*.fits' -exec mv '{}' ./Ancillary/A/bis/ \;

#Organize B data
find ./Ancillary/data/reduced/ -type f -iname '*ccf*B*.fits' -exec mv '{}' ./Ancillary/B/ccf/ \;

find ./Ancillary/data/reduced/ -type f -iname '*e2ds*B*.fits' -exec mv '{}' ./Ancillary/B/e2ds/ \;

find ./Ancillary/data/reduced/ -type f -iname '*s1d*B*.fits' -exec mv '{}' ./Ancillary/B/s1d/ \;

find ./Ancillary/data/reduced/ -type f -iname '*ccf*B*.tbl' -exec mv '{}' ./Ancillary/B/tables/ \;

find ./Ancillary/data/reduced/ -type f -iname '*bis*B*.fits' -exec mv '{}' ./Ancillary/B/bis/ \;

#Organize INT_GUIDE data
find ./Ancillary/data/reduced/ -type f -iname '*INT_GUIDE*.fits' -exec mv '{}' ./Ancillary/intGuide/ \;

#Organize Spectra
find ./archive/ -type f -iname '*.fits' -exec mv '{}' ./Spectra/ \;

#Rename ./archive/ folder
mv ./archive/ ./README/

#Remove data folder if there are no files left
if [[ $(find ./Ancillary/data/ -type f) ]]
then echo 'There are files left'
else rm -rf ./Ancillary/data/
fi
