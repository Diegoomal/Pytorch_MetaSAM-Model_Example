
echo "===== bash build ====="

echo "1) Reset environment"

# source pipeline/destroy.sh

echo "2) Conda environment"

conda deactivate

conda env create -n project-env -f ./env.yml

conda activate project-env

echo "3) Install SAM"

git clone https://github.com/facebookresearch/segment-anything.git

cd segment-anything

pip install -e .

cd ..

echo "4) Copy dataset"

cd src/

mkdir images

wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

cd ..

echo "5) LINT verify with Flak8"

flake8 . --count --statistics

flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

echo "6) Generating documentation"

# pdoc src -o docs

echo "7) Run project"

python src/main.py

echo "8) Unity test with pytest"

pytest -s