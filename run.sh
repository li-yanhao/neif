#!/usr/bin/env bash

set -eu

bins=16
w=8
th=3
search_range=5
quantile=5
demosaic=""
prec_lvl=3
add_noise=""
noise_a="0.2"
noise_b="0.2"
post_correction=""

while getopts :b:w:t:s:q:d:p:N:A:B:c: option
do 
    case "${option}" in
        b) bins=${OPTARG};;
        w) w=${OPTARG};;
        t) th=${OPTARG};;
        s) search_range=${OPTARG};;
        q) quantile=${OPTARG};;
        d)  if [ ${OPTARG} = "True" ]
            then
                demosaic="-demosaic"
            elif [ ${OPTARG} = "False" ]
            then
                demosaic=""
            else
                echo "Error: demosaic option should be True or False"
                exit 1
            fi
            ;;
        p) prec_lvl=${OPTARG};;
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
        c) if [ ${OPTARG} = "True" ]
            then
                post_correction="-post_correction"
            elif [ ${OPTARG} = "False" ]
            then
                post_correction=""
            else
                echo "Error: post_correction option should be True or False"
                exit 1
            fi
            ;;
            
            
        
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
# out_curve="curve.png"

#####################
#      IPOL env     #
#####################
main=$bin/neif/main.py

#####################
#   Main execution  #
#####################
command="python $main $img_0 $img_1 $out_curve \
    -bins $bins \
    -quantile $quantile  \
    -w $w \
    -th $th \
    -search_range $search_range \
    -prec_lvl $prec_lvl \
    -noise_a $noise_a \
    -noise_b $noise_b \
    $post_correction \
    $add_noise \
    $demosaic"

echo $command
$command


# echo "Configuration done!"
