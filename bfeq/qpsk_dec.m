% QPSK decoder for SOQPSK from MIL-STD-188-181C
%
% Input symbols are in a complex array at one sample per symbol. If soft_dec = 1
% then soft information is returned and the outputs are floating point. When
% soft_dec = 0, hard decisions are made and the outputs are logicals.

function bits = qpsk_dec(syms, soft_dec)

    % Soft decisions
    if soft_dec
        
        % Return unquantized I and Q symbols
        i_bits = real(syms);
        q_bits = imag(syms);
        
    % Hard decisions
    else
        
        % Symbol values of +1 map to bit values of 1
        % Symbol values of -1 map to bit values of 0
        i_bits = real(syms) > 0;
        q_bits = imag(syms) > 0;
        
    end
    
    % First bit received is from the I channel; output is [I, Q, I, Q, ... ]
    bits = [i_bits, q_bits];
    bits = reshape(bits', numel(bits), 1);
    
end