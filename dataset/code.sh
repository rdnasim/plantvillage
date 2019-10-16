#!/bin/bash
dir="/home/rdnasim/Workspace/uni-thesis/plantvillage/dataset/p/"
d=$("ls $dir")
num=1
for i in $d
do
num=$(($num+1))
mv $i $num.JPG
done