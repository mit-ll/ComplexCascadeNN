% QPSK encoder for SOQPSK from MIL-STD-188-181C
%
% Input is a column vector of logicals
% Output is a column vector of complex symbols

function syms = qpsk_enc(bits)

    % Input error checking
    if ~islogical(bits)
        error('Input bits must be logicals');
    end
    if mod(length(bits), 2)
        error('Number of input bits must be even');
    end

    % Separate into I and Q channels
    i_bits = bits(1:2:end);
    q_bits = bits(2:2:end);
    
    % Map bit 1 to symbol +1 and bit 0 to symbol -1
    i_syms = 2*i_bits - 1;
    q_syms = 2*q_bits - 1;
    
    % Complex output
    syms = i_syms + 1j*q_syms;

end