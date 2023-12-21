# Installation instructions
1. create conda environment 
```
conda create -n neurosim
conda activate neurosim
```

2. Install PyTorch
go to https://pytorch.org/
select correct OS, Conda and the proper compute platform (we have CUDA 12.2 drivers so we select CUDA 12.1)
paste the command into your terminal

3. Verify dependencies for lm harness
```
cd lm-evaluation-harness
/path/to/conda/envs/neurosim/bin/pip install -e .
```

4. Verify dependencies for TensorRT
```
cd pytorch-quantization
/path/to/conda/envs/neurosim/bin/pip install -e .
```

5. Install a few more packages
```
conda install -c conda-forge matplotlib sentencepiece protobuf
```

6. Make sure LD_LIBRARY_PATH includes conda directory
```
export LD_LIBRARY_PATH=/path/to/conda/envs/neurosim/lib/:$LD_LIBRARY_PATH
```

7. OPTIONAL: create new cache directory if you don't want to use /nethome/USER/cache

```
mkdir /path/to/cache/
export HF_DATASETS_CACHE="/path/to/cache/"
```

# Run example 
Calibrate quantizers for inputs, weights, ADC outputs and run inference evaluation
1. Download checkpoint of llama-2
2. cd lm-evaluation-harness
3. Change directory of pretrained llama-2 on line 160 in main.py
4. bash autorun.sh
5. Check output.out and results.csv for the output
