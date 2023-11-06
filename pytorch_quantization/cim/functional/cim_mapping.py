import torch
import math
import numpy as np

def convert_to_n_ary(dec_matrix, base, bits=8, device='cuda'):
    # expand each column in the decimal matrix to an n-ary number
    rows, cols = dec_matrix.shape
    dec_matrix = dec_matrix.flatten().reshape(-1,1).int()

    max_val = 2**bits
    num_digits = math.ceil(math.log(max_val, base))

    n_ary = base**torch.arange(num_digits, device=device).flip(0)

    out = dec_matrix // n_ary % base

    return out.reshape(rows, num_digits*cols)

def map_weights(args, weights, original_shape):
    rows_PE = math.ceil(weights.shape[0] / args.sub_array[0])
    cols_PE = math.ceil(weights.shape[1] / args.sub_array[1])

    with open('PE_sizes.txt', 'a') as f:
        f.write(f'weight shape: {original_shape[0]}x{original_shape[1]} converted shape: {weights.shape[0]}x{weights.shape[1]} PE shape: {rows_PE}x{cols_PE}\n')

def print_network_info(args, input_shape, weight_shape, stride):

    with open('NetWork_'+args.model_name+'.csv', 'a') as f:

        if len(weight_shape) < 4:
            f.write(f'1,1,{input_shape[1]},1,1,{weight_shape[0]},0,1\n')

        else:
            f.write(f'{input_shape[2]},{input_shape[3]},{input_shape[1]},{weight_shape[2]},{weight_shape[3]},{weight_shape[0]},0,{stride[0]}\n')
