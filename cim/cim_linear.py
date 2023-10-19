import torch
from torch.nn.functional import unfold
from . import cim_sim
from .cim_mapping import print_network_info

def cim_linear(cim_args, quant_input, scale_input, quant_weight, scale_weight, bias=None):
    """
    Performs a matrix multiplication followed by the addition of a bias vector.

    Arguments:
        input (torch.Tensor): Input tensor of shape (N, C_in).
        weight (torch.Tensor): Weight tensor of shape (C_out, C_in).
        bias (torch.Tensor): Bias tensor of shape (C_out).

    Returns:
        torch.Tensor: Output tensor of shape (N, C_out).
    """

    input_shape = quant_input.shape
    weight_shape = quant_weight.shape

    if len(input_shape) > 2:
        quant_input = quant_input.flatten(start_dim=0, end_dim=-2)


    if cim_args.hardware==True:
        print_network_info(cim_args, quant_input.shape, weight_shape, 1)

    # Convert inputs and weights to integers
    quant_input = quant_input * scale_input
    quant_weight = quant_weight * scale_weight

    scale = (scale_input * scale_weight).view(1,-1)

    # Shift inputs and weights so they are positive
    shift_input = -torch.min(quant_input)
    shift_weight = -torch.min(quant_weight)
    quant_input = quant_input + shift_input
    quant_weight = quant_weight + shift_weight

    # save input tensor as a .bin file
    # quant_input = quant_input.reshape(-1, quant_input.shape[1])
    # quant_input = quant_input.to(dtype=torch.int32)
    # torch.save(quant_input, 'input_matrix.bin')
    
    # Add a dummy row to the input matrix
    quant_input = torch.cat((quant_input, torch.full((1,quant_input.shape[1]), fill_value=shift_input, device=cim_args.device)), dim=0)

    # Reshape the weight tensor into a matrix
    quant_weight = quant_weight.view(weight_shape[0], -1).t()

    # save input tensor as a .bin file
    # quant_weight = quant_weight.to(torch.int32)
    # torch.save(quant_weight, 'weight_matrix.bin')

    # save the output as a .bin file
    # quant_output = torch.matmul(quant_input, quant_weight)
    # torch.save(quant_output, 'output_matrix.bin')

    # Add a dummy column to the weight matrix
    quant_weight = torch.cat((quant_weight, torch.full((quant_weight.shape[0],1), fill_value=shift_weight, device=cim_args.device)), dim=1)

    # output_correct = torch.matmul(quant_input, quant_weight)
    
    # Perform matrix multiplication on CIM crossbar
    quant_input = quant_input.to(torch.int32)     # THIS PART HAS SOME LOSS
    quant_weight = quant_weight.to(torch.int32)   # THIS PART HAS SOME LOSS


    output = cim_sim.simulate_array(cim_args, quant_input, quant_weight)

    # compare outputs
    # diff = torch.abs(output_correct - output)
    # total_loss = torch.sum(diff)
    # print(total_loss) 

    out_dummy_row = output[-1,:-1].unsqueeze(0) # extract dummy row
    out_dummy_col = output[:-1,-1].unsqueeze(1) # extract dummy column
    shift = output[-1,-1]                       # extract dummy element

    output = output[:-1,:-1] # remove dummy row and column   

    # Remove shifts
    output = output - out_dummy_row - out_dummy_col + shift

    # Rescale
    output = output/scale

    # Reshape output
    output_shape = input_shape[:-1] + weight_shape[0:1]
    output = output.reshape(output_shape)

    # Add bias if provided
    if bias is not None:
        output += bias
    
    return output
