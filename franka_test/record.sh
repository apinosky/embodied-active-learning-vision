#!/bin/bash

seed=0010
mod=_xyw_plant_duck

for sensor in rgb # intensity
do
    for explr in  entklerg # uniform
    do
        for base_path in data # sim_data
        do
            ### playback trajectory
            playback_path="$base_path/$sensor/${explr}_${seed}${mod}/"
            echo $playback_path
            # python generate_traj_video.py $playback_path
            ### save video
            for vid in  franka_traj explr
            do
                path="$base_path/$sensor/${explr}_${seed}${mod}/${vid}"
                file_name=${sensor}_${explr}_${seed}${mod}_${vid}
                pushd $path
                # gifski --fps 7 -o ${file_name}.gif *.svg # to make gifs
                ffmpeg -r:v 50 -i iter%05d.svg  -codec:v libx264 -preset veryslow -pix_fmt yuv420p -crf 28 -an ${file_name}.mp4 # to make mp4s
                # ffmpeg -i ${file_name}.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ${file_name}.mp4 # to make video from gif
                popd
            done
        done
    done
done

## extra stuff for saving various other files
## ID
for base_name in "explr_1000steps_overview"  "explr_1000steps_heatmaps_L2_1.0_mean" "explr_1000steps_heatmaps_L2_1.0_max"; do ffmpeg -framerate 0.1  -pattern_type glob -i "${base_name}_step*.svg" -codec:v libx264  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -preset veryslow -pix_fmt yuv420p -crf 28 -an ${base_name}.mp4; done

## fingerprints
# for base_name in "fp_id0"  "fp_id1" "fp_id2" "fp_id3"; do ffmpeg -r:v 1.25 -pattern_type glob -i "${base_name}_eval*.svg" -codec:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -preset veryslow -pix_fmt yuv420p -crf 28 -an ${base_name}.mp4; done

## only zip mp4 files
# zip -r videos.zip . -i '*.mp4'
