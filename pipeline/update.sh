echo "1) Conda environment"

conda deactivate

conda env update -n project-env -f ./env.yml

conda activate project-env

echo "2) Install SAM"

pip uninstall segment_anything

rm -rf segment-anything

git clone https://github.com/facebookresearch/segment-anything.git

cd segment-anything

pip install -e .

cd ..

echo "3) Run project"

python src/main.py

echo "4) Unity test with pytest"

pytest -s