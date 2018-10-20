#!/bin/bash

echo "CONVERTING videos in ${VIDEO_BASE_DIR}"

AD_DIR=ad
GAME_DIR=game

DIRS=($AD_DIR $GAME_DIR)

for dir in ${DIRS[@]}; do
  echo "converting $dir"

  for video_file in $VIDEO_BASE_DIR/$dir/*.webm; do
    echo "converting $video_file"
    outfile=$(basename $video_file ".webm")
    ffmpeg -i $video_file -vf "scale=320:240, fps=$FPS, format=gray" $DESTINATION_DIR/$dir-$outfile-%d.jpg > $LOG_DIR/conversion.log 2>&1
  done
done
