#!/usr/bin/bash

# LEFT_NUM=1
# LAST_NUM=291
# WKSP=
# DATA_DIR=""
# WORK_DIR=""
# WELL=W0001
# FIELD=F0001

INPUT_IMAGE_FILE_MASK_EXT=".tif"

INPATH="$DATA_DIR"/$WELL$FIELD
OUTPATH="$WORK_DIR"/$WELL$FIELD

DD=`date`
echo $DD >> done_komet.log
echo LEFT_NUM=$LEFT_NUM >> done_komet.log
echo LAST_NUM=$LAST_NUM >> done_komet.log
echo WKSP=$WKSP >> done_komet.log
echo DATA_DIR=$DATA_DIR >> done_komet.log
echo WORK_DIR=$WORK_DIR >> done_komet.log
echo $WELL >> done_komet.log
echo $FIELD >> done_komet.log

TOUCH_FILE=done_komet
LOG_FILE=done_komet.log

if [ ! -f $TOUCH_FILE ]
    then
    KMD="komet -L $LAST_NUM -w -p kozlov_kn@spbstu.ru,$OUTPATH,$INPATH,localhost:7778 $WKSP"
    echo $KMD >> $LOG_FILE
    komet -L $LAST_NUM -w -p kozlov_kn@spbstu.ru,"$OUTPATH","$INPATH",localhost:7778 "$WKSP"
    touch $TOUCH_FILE
fi

DATA_DIR=${DATA_DIR// /\\ }
WORK_DIR=${WORK_DIR// /\\ }

INPATH="$DATA_DIR"/$WELL$FIELD
OUTPATH="$WORK_DIR"/$WELL$FIELD

BNWKSP=`basename $WKSP`

for t in $(seq $LEFT_NUM $LAST_NUM);
    do
        if [ "$t" -lt 10 ]; then
            tt="000$t"
        elif [ "$t" -lt 100 ]; then
            tt="00$t"
        else
            tt="0$t"
        fi
        TEST_IMAGE_FILE=$INPATH/"$WELL$FIELD"T"$tt"Z001C1$INPUT_IMAGE_FILE_MASK_EXT
        echo $TEST_IMAGE_FILE
        if [ -f "$TEST_IMAGE_FILE" ]
    	    then
    	    TEST_FILE="$WELL$FIELD"T"$tt"Z001C1_tab.csv
    	    WKSP_FILE="$WELL$FIELD"T"$tt"Z001C1_$BNWKSP
    	    while [ ! -f $TEST_FILE ];
        	do
            	    KMD="komet $WKSP_FILE"
                    echo $KMD >> $LOG_FILE
	            komet "$WKSP_FILE"
    		done
    	fi
done

REG_KMD=/mnt/wdb4/data/colony_tracking/colonies_tracking_base/ColoniesTracker/ImageRegistrationCore.py
TRK_KMD=/mnt/wdb4/data/colony_tracking/colonies_tracking_base/ColoniesTracker/CoreTrackingAPI.py

TOUCH_FILE_REG=done_reg

if [ ! -f $TOUCH_FILE_REG ]
    then
    echo tag_sym="11,15" \
    last_num=$LAST_NUM \
    viewer=OFF \
    features_cnt=15000 \
    N_CORES=2 \
    parallel=OFF \
    file_mask=$INPUT_IMAGE_FILE_MASK_EXT \
    reg_algo=FFT >> $LOG_FILE
    python3 $REG_KMD \
    "$INPATH" \
    "$OUTPATH"/"$WELL$FIELD"_reg \
    tag_sym="11,15" \
    last_num=$LAST_NUM \
    viewer=OFF \
    features_cnt=15000 \
    N_CORES=2 \
    parallel=OFF \
    reg_algo=FFT \
    file_mask=$INPUT_IMAGE_FILE_MASK_EXT
#    if [ -f "$OUTPATH"/"$WELL$FIELD"_reg.txt ]
    if [ -f "$WELL$FIELD"_reg.txt ]
    then
        touch $TOUCH_FILE_REG
    fi
fi

max_dist_param=20
max_gap=6
split_merge=True
min_track_len=10
min_tree_len=50
merging_cost_cutoff_multiplier=1.25
frames_cnt=-1
sq_lower_bound=499
sq_upper_bound=999999

echo ffile_mask="_tab.csv" \
max_dist_param=$max_dist_param \
max_gap=$max_gap \
split_merge=$split_merge \
min_track_len=$min_track_len \
min_tree_len=$min_tree_len \
merging_cost_cutoff_multiplier=$merging_cost_cutoff_multiplier \
frames_cnt=$frames_cnt \
sq_lower_bound=$sq_lower_bound \
sq_upper_bound=$sq_upper_bound \
bkg_img="$OUTPATH"/"$WELL$FIELD"T0001Z001C1_movl.jpg \
tag_sym="11,15" \
napari_viewer="OFF" \
custom_offset_mode="OFF" \
matplot_lib="ON" \
shifts_path="$OUTPATH"/"$WELL$FIELD"_reg.txt \
save_image_path="$OUTPATH"/"$WELL$FIELD"_reg_tracks.png >> $LOG_FILE

python3 $TRK_KMD \
"$OUTPATH" \
"$OUTPATH"/work_trk \
file_mask="_tab.csv" \
max_dist_param=$max_dist_param \
max_gap=$max_gap \
split_merge=$split_merge \
min_track_len=$min_track_len \
min_tree_len=$min_tree_len \
merging_cost_cutoff_multiplier=$merging_cost_cutoff_multiplier \
frames_cnt=$frames_cnt \
sq_lower_bound=$sq_lower_bound \
sq_upper_bound=$sq_upper_bound \
bkg_img="$OUTPATH"/"$WELL$FIELD"T0001Z001C1_movl.jpg \
tag_sym="11,15" \
napari_viewer="OFF" \
custom_offset_mode="OFF" \
matplot_lib="ON" \
shifts_path="$OUTPATH"/"$WELL$FIELD"_reg.txt \
save_image_path="$OUTPATH"/"$WELL$FIELD"_reg_tracks.png
