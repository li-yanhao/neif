#!/usr/bin/env bash

set -eu

bins=16
w=8
th=3
search_range=10
quantile=5
demosaic=""
auto_quantile="manual"

while getopts :b:w:t:s:q:d:a: option
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
        a)  if [ ${OPTARG} = "auto" ]
            then
                auto_quantile="-auto_quantile"
            elif [ ${OPTARG} = "manual" ]
            then
                auto_quantile=""
            else
                echo "Error: auto_quantile option should be True or False"
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

#####################
#      IPOL env     #
#####################
main=$bin/neif/main.py

#####################
#   Main execution  #
#####################
python $main $img_0 $img_1 $out_curve \
    -bins $bins \
    -quantile $quantile \
    -w $w \
    -th $th \
    -search_range $search_range \
    $demosaic \
    $auto_quantile


# echo "Configuration done!"
