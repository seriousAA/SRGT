#!/usr/bin/env bash

# Define source and destination directories
SOURCE_DIR="/home/****/RS-PCT/data/DOTA/gsd_after"
DEST_DIR="/home/****/RS-PCT/data/DOTA/gsd_dataset/test"

# List of images to copy
images=(
"P0717_0"
"P1741_3"
"P1332_15"
"P1735_20"
"P1530_23"
"P1623_20"
"P1003_81"
"P1528_31"
"P1303_0"
"P0743_5"
"P1769_12"
"P1621_21"
"P1611_36"
"P0333_7"
"P1690_3"
"P1212_22"
"P1646_12"
"P1653_13"
)

# Loop through the images and copy each one
for image in "${images[@]}"; do
  echo "Copying ${image}.png to $DEST_DIR"
  cp "${SOURCE_DIR}/${image}.png" "$DEST_DIR"
done

echo "Copy process completed."
