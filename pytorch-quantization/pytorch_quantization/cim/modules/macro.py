import torch    
import math

class CIM():
    def simulate_array(self, input2d, weight2d):
        # Shift inputs and weights so they are positive
        shift_input  = -torch.min(input2d)
        shift_weight = -torch.min(weight2d)
        input2d = input2d + shift_input
        weight2d = weight2d + shift_weight        

        # Add a dummy row to the input matrix
        input2d = torch.cat((input2d, torch.full((1,input2d.shape[1]), fill_value=shift_input, device=input2d.device)), dim=0)

        # Add a dummy column to the weight matrix
        weight2d = torch.cat((weight2d, torch.full((weight2d.shape[0],1), fill_value=shift_weight, device=weight2d.device)), dim=1)

        # Perform matrix multiplication on CIM crossbar
        input2d = input2d.to(torch.int32)     # THIS PART HAS SOME LOSS
        weight2d = weight2d.to(torch.int32)   # THIS PART HAS SOME LOSS

        # convert the integer weights to n_ary representation
        weight2d = convert_to_n_ary(weight2d, base=len(self._cim_args.mem_values), bits=self._cim_args.weight_precision, device=self._cim_args.device)
       
        # convert weights to eNVM cell values if running hardware simulation
        if self._cim_args.hardware:
            weight2d = self.convert_weights(weight2d)   

        # divide the weight matrix into partitions
        num_partitions = math.ceil(weight2d.shape[0]/self._cim_args.open_rows) 
        ADC_out = torch.zeros((input2d.shape[0], weight2d.shape[1]), device=self._cim_args.device)
        Psum = torch.zeros_like(ADC_out)

        # calculate outputs for each bit of the input
        for i in range(self._cim_args.input_precision):
            mask = 2**i
            input = (input2d & mask) >> i

            # calculate partial sum for each partition
            # TODO: do this loop in parallel
            Psum[:,:] = 0

            for part in range(num_partitions):  

                start_row = part*self._cim_args.open_rows
                end_row   = start_row + self._cim_args.open_rows

                if self._cim_args.hardware:
                    # simulate analog MAC and ADC operation
                    out = self.ADC_output(input[:, start_row:end_row], weight2d[start_row:end_row, :])

                else:
                    out = torch.matmul(input[:, start_row:end_row].to(dtype=torch.float32), weight2d[start_row:end_row, :].to(dtype=torch.float32))

                    # quantize adc output
                    out = self._adc_quantizers[part](out)

                # add partition output to total output of the sub array
                Psum += out

            # scale partial sum for input bit significance
            Psum *= mask

            # add partition output to total output of the sub array
            ADC_out += Psum

        max_val = 2**self._cim_args.weight_precision
        base = len(self._cim_args.mem_values)
        cols_per_weight = math.ceil(math.log(max_val, base))
        weights_mask = base**torch.arange(cols_per_weight, device=self._cim_args.device).flip(0)

        # split output into groups of each dot product
        ADC_out = ADC_out.reshape(ADC_out.shape[0], int(ADC_out.shape[1]/cols_per_weight), cols_per_weight)

        # multiply each dot product by the weight mask
        ADC_out *= weights_mask

        # add output bits together and accumulate total output
        ADC_out = ADC_out.sum(dim=-1)

        out_dummy_row = ADC_out[-1,:-1].unsqueeze(0) # extract dummy row
        out_dummy_col = ADC_out[:-1,-1].unsqueeze(1) # extract dummy column
        shift = ADC_out[-1,-1]                       # extract dummy element

        ADC_out = ADC_out[:-1,:-1] # remove dummy row and column   

        # Remove shifts
        return ADC_out - out_dummy_row - out_dummy_col + shift        

    def convert_weights(self, n_ary):
        weights = self._cim_args.mem_values[n_ary]

        # augment weights with programming noise, if specified
        if self._cim_args.weight_noise is not None:
            weights += self._cim_args.weight_noise

        # augment weights with resistance drift
        if self._cim_args.stress_time > 0:
            weights += self.calc_drift(self._cim_args.stress_time)
        
        if self._cim_args.crossbar_type == 'resistive':
            # if we are using resistive crossbar, convert to conductance
            return 1/weights
        elif self._cim_args.crossbar_type == 'capacitive':
            # if we are using capacitive crossbar, keep capacitance
            return weights

    def calc_drift(self, stress_time):
        # TODO: match this with NeuroSim V1.4
        return 0       

    def ADC_output(self, inputs, weights):
        
        inputs = inputs.to(dtype=torch.float32)

        equiv_cond = torch.matmul(inputs, weights)

        del weights
        del inputs

        # # subtract memory IR drop
        # vdd = self._cim_args.vdd - self._cim_args.mem_IR_drop

        vdd = self._cim_args.vdd

        if self._cim_args.conversion_type == 'TIA':
            # use trans-impedence amplifier to convert current to voltage
            # BL_voltages = torch.mul(torch.div(vdd, equiv_res), Rf)
            BL_voltages = torch.mul(equiv_cond, vdd*self._cim_args.Rf)
        elif self._cim_args.conversion_type == 'PU': # converting current sum to voltage using pull-up PMOS
            print("ERROR: PU conversion type not supported yet")
            BL_voltages = vdd - equiv_cond*self._cim_args.res_divider
            exit(1)
            # min = torch.min(BL_voltages)
            # if (min < 0):
            #     print(min)
        else:
            # use voltage divider to convert current to voltage
            BL_voltages = torch.div(vdd, 1 + torch.mul(equiv_cond, (self._cim_args.res_divider + self._cim_args.Rpdn)))

        if self._cim_args.v_noise > 0:
            noise = torch.normal(mean=0.0, std=self._cim_args.v_noise, size=BL_voltages.shape, device=self._cim_args.device)

            # add to BL voltages depending on std of nosie
            BL_voltages += noise

        if self._cim_args.analog_shift_add == True:
            # shift BL voltages and add them together

            # create a mask for the weight precision
            max_val = 2**self._cim_args.weight_precision
            base = len(self._cim_args.mem_values)
            cols_per_weight = math.ceil(math.log(max_val, base))
            weights_mask = base**torch.arange(cols_per_weight, device=self._cim_args.device).flip(0)

            # split output into groups of each dot product
            BL_voltages = BL_voltages.reshape(BL_voltages.shape[0], int(BL_voltages.shape[1]/cols_per_weight), cols_per_weight)

            # multiply each dot product by the weight mask
            BL_voltages *= weights_mask        

            # sum outputs together
            BL_voltages = BL_voltages.sum(dim=-1)

        return self.sense_voltage(BL_voltages)

    def sense_voltage(self, BL_voltages):

        num_refs = 2**self._cim_args.adc_precision
        BL_voltages = BL_voltages.contiguous()

        if self._cim_args.conversion_type == 'PU':
            ADC_output = num_refs - torch.bucketize(BL_voltages, self._cim_args.v_ref.flip(0), out_int32=True, right=True)

        elif self._cim_args.conversion_type == 'TIA':
            ADC_output = torch.bucketize(BL_voltages, self._cim_args.v_ref, out_int32=True, right=True)

        return ADC_output

def convert_to_n_ary(dec_matrix, base, bits=8, device='cuda'):
    # expand each column in the decimal matrix to an n-ary number
    rows, cols = dec_matrix.shape
    dec_matrix = dec_matrix.flatten().reshape(-1,1).int()

    max_val = 2**bits
    num_digits = math.ceil(math.log(max_val, base))

    n_ary = base**torch.arange(num_digits, device=device).flip(0)

    out = dec_matrix // n_ary % base

    return out.reshape(rows, num_digits*cols)