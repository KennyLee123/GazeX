#!/bin/bash

conda create -n gazex python=3.10.0 -y
conda activate gazex
pip install -r gazex_requirements.txt
