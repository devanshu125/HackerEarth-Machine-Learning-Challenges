#!/bin/bash

for i in {1..1}; 
do 
    python -W ignore clean_together.py
    echo "Cleaning done..."
    python create_folds.py
    echo "Folds created..."
    python lbl_tree.py --model1 "lgbm" --model2 "cb"
done 

echo "Press any key to continue"
while [ true ] ; do
read -t 3 -n 1
if [ $? = 0 ] ; then
exit ;
else
echo "waiting for the keypress"
fi
done