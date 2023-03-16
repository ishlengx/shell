#!/bin/bash
WARNING_COLOR="echo -e \E[1;31m"
END="\E[0m"
DIR=/tmp/`date +%F_%T`
mkdir $DIR
mv $* $DIR
${WARNING_COLOR} Move $* to $DIR $END