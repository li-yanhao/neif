#!/usr/bin/env bash

set -eu

bins=16
w=8
T=9
th=3
search_range=5
quantile=0.01
multiscale=-1
subpx_order=0
# add_noise="-add_noise"
add_noise=""
noise_a="0.2"
noise_b="0.2"
grayscale=""



while getopts :b:w:T:t:s:q:g:m:S:N:A:B: option
do 
    case "${option}" in
        b) bins=${OPTARG};;
        w) w=${OPTARG};;
        T) T=${OPTARG};;
        t) th=${OPTARG};;
        s) search_range=${OPTARG};;
        q) quantile=${OPTARG};;
        g)  if [ ${OPTARG} = "True" ]
            then
                grayscale="-grayscale"
            elif [ ${OPTARG} = "False" ]
            then
                grayscale=""
            else
                echo "Error: grayscale option should be True or False"
                exit 1
            fi
            ;;
        m)  multiscale=${OPTARG};;
        S)  subpx_order=${OPTARG};;
        N)  if [ ${OPTARG} = "True" ]
            then
                add_noise="-add_noise"
            elif [ ${OPTARG} = "False" ]
            then
                add_noise=""
            else
                echo "Error: add_noise option should be True or False"
                exit 1
            fi
            ;;
        A) noise_a=${OPTARG};;
        B) noise_b=${OPTARG};;
            
            
        
    esac
done

img_0=${@:$OPTIND:1}
img_1=${@:$OPTIND+1:1}
out_curve=${@:$OPTIND+2:1}


#####################
# TEST in local env #
#####################
# main=./main.py
# img_0="frame0.png"
# img_1="frame1.png"

# img_0="/Users/yli/phd/ipol_noise_estimation/room_0.png"
# img_1="/Users/yli/phd/ipol_noise_estimation/room_1.png"

# img_0="/Users/yli/phd/ipol_noise_estimation/room_0.png"
# img_1="/Users/yli/phd/ipol_noise_estimation/room_1.png"

# img_0="/Users/yli/phd/ipol_noise_estimation/ens_0.png"
# img_1="/Users/yli/phd/ipol_noise_estimation/ens_1.png"


#####################
#      IPOL env     #
#####################
main=$bin/neif/main.py

#####################
#   Main execution  #
#####################
command="python $main $img_0 $img_1 \
    -bins $bins \
    -quantile $quantile  \
    -w $w \
    -T $T \
    -th $th \
    -search_range $search_range \
    -noise_a $noise_a \
    -noise_b $noise_b \
    -multiscale $multiscale \
    -subpx_order $subpx_order \
    $add_noise \
    $grayscale"

# echo $command
$command

