# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Clone DRAGIN and set it up
git clone git@github.com:oneal2000/DRAGIN.git
cd DRAGIN
conda create -n dragin python=3.9
conda activate dragin
pip install torch==2.1.1
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Copy our files appropriately
cp ../code/src/*.py ./src/
cp ../code/create_elastic.py .
cp ../code/prep_elastic.py .
rm -rf ./config
cp ../code/config .
