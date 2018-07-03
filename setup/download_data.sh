mkdir data
python ./setup/download.py mnist

wget https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat
mkdir data/omniglot
mv chardata.mat data/omniglot/
python ./setup/preprocess_omniglot.py

python ./setup/download.py celebA
cd data/celebA
xargs rm < ./../../setup/celebA_unused.txt