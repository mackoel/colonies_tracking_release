#!/usr/bin/bash

# LEFT_NUM=1
# LAST_NUM=291
# WKSP=
# DATA_DIR=""
# WORK_DIR=""
# WELL=W0001
# FIELD=F0001

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
    # $KMD
fi

touch $TOUCH_FILE

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
        TEST_FILE="$WELL$FIELD"T"$tt"Z001C1_tab.csv
        WKSP_FILE="$WELL$FIELD"T"$tt"Z001C1_$BNWKSP
        while [ ! -f $TEST_FILE ];
            do
                KMD="komet $WKSP_FILE"
                echo $KMD >> $LOG_FILE
                komet "$WKSP_FILE"
        done
done

REG_KMD=/mnt/wdb4/data/colony_tracking/colonies_tracking_base/ColoniesTracker/ImageRegistrationCore.py
TRK_KMD=/mnt/wdb4/data/colony_tracking/colonies_tracking_base/ColoniesTracker/CoreTrackingAPI.py

python3 $REG_KMD \
$INPATH \
$OUTPATH/"$WELL$FIELD"_reg \
tag_sym="11,15" \
last_num=$LAST_NUM \
viewer=OFF \
features_cnt=15000 \
N_CORES=2 \
parallel=OFF \
reg_algo=FFT

python3 $TRK_KMD \
$OUTPATH \
$OUTPATH/work_trk \
ffile_mask="_tab.csv" \
max_dist_param=35 \
max_gap=4 \
split_merge=True \
min_track_len=1 \
min_tree_len=50 \
merging_cost_cutoff_multiplier=1.0 \
frames_cnt=-1 \
sq_lower_bound=99 \
sq_upper_bound=999999 \
bkg_img=$OUTPATH/"$WELL$FIELD"T0001Z001C1_movl.jpg \
tag_sym="11,15" \
napari_viewer="OFF" \
custom_offset_mode="OFF" \
matplot_lib="ON" \
split_merge=False \
shifts_path=$OUTPATH/"$WELL$FIELD"_reg.txt \
save_image_path=$OUTPATH/"$WELL$FIELD"_reg_tracks.png
