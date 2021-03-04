% SOQPSK modulator from MIL-STD-188-181C (pages 24-26)
%
% Input bits are a logical column vector
% sps is the number of samples per symbol
% alpha is the shaping factor (related to how fast symbol transitions occur)

function x = soqpsk_mod(bits, sps, alpha)

    % --------------------------------------------------------------------------
    % Setup
    % --------------------------------------------------------------------------

    % Default is 4 samples/symbol and alpha = 0.5
    if nargin < 2
        sps = 4;
        alpha = 0.5;
    elseif nargin < 3
        alpha = 0.5;
    end

    % Input error checking
    if ~islogical(bits)
        error('Input bits must be logicals');
    end
    if mod(sps, 4)
        error('Samples per symbol must be a multiple of 4');
    end
    if (alpha <= 0) || (alpha > 0.5)
        error('Alpha must be > 0 and <= 0.5');
    end
    
    % Define the phase transition properties
    t_start = (1 - alpha)*(sps/2);      % Start of the first transition
    t_len = alpha*sps;                  % Transition length
    t_hold = (sps/2) - t_len;           % Hold time after transition
    
    % Phase increment. Each transition between OQPSK symbols is pi/2 radians,
    % and the transition time is defined above as t_len. pi/2 is sps/2, and this
    % simplifies to 1/(2*alpha)
    p_inc = (sps/2)/t_len;
    
    % Ensure transition points and phase increment are integers
    if t_start ~= round(t_start)
        error('Transition start is not an integer: %g', t_start)
    elseif t_len ~= round(t_len)
        error('Transition length is not an integer: %g', t_len)
    elseif t_hold ~= round(t_hold)
        error('Hold time is not an integer: %g', t_hold)
    elseif p_inc ~= round(p_inc)
        error('Phase increment is not an integer: %g', p_inc);
    end
    
    % Lookup table - one period of a complex sinusoid at 2x the sample rate
    % (0 to pi is sps samples, so 0 to 2*pi is 2*sps samples)
    dp = pi/sps;
    p = 0:dp:(2*pi - dp);
    phase_lut = cos(p) + 1j*sin(p);
    n_lut = length(phase_lut);
    
    % Initial state on the I and Q channels. We send the first bit on the I
    % channel, and transmit a bit value of 1 on the Q channel until the first
    % bit arrives on the Q channel.
    i_state = bits(1);
    q_state = 1;
    
    % Preallocate the output
    n_bits = length(bits);
    n_samps = ceil(n_bits/2)*sps;
    if ~mod(n_bits, 2)
        n_samps = n_samps + sps/2;
    end
    x = zeros(n_samps, 1);
    out_idx = 1;
    
    % --------------------------------------------------------------------------
    % Processing
    % --------------------------------------------------------------------------
    
    % Initial state of the phase accumulator - determined by the first bit on
    % the I channel, as the first bit on the Q channel is always a 1.
    if i_state == 1
        p_acc = sps/4;      % pi/4, symbol = +1 + 1j
    else
        p_acc = 3*sps/4;    % 2*pi/4, symbol = -1 + 1j
    end
    
    % Send the inital phase until the first possible transition on the Q channel
    for i = 1:t_start
        x(out_idx) = phase_lut(p_acc+1);
        out_idx = out_idx + 1;
    end
    
    % Send all remaining bits
    for i = 2:n_bits
        
        % Even bits --> Q channel
        if ~mod(i, 2)
            
            % Only change phase if the state has changed
            q_bit = bits(i);
            if q_bit ~= q_state
                
                % When I and Q are in the same state and Q is changing, the
                % phase can only decrease. Otherwise, if I and Q are not in the
                % same state and Q is changing, the phase can only increase.
                if q_state == i_state
                    s = -1;
                else
                    s = 1;
                end
                
                % Transition the phase, ensuring it wraps in the lookup table
                for j = 1:t_len
                    p_acc = p_acc + s*p_inc;
                    p_acc = mod(p_acc, n_lut);
                    x(out_idx) = phase_lut(p_acc+1);
                    out_idx = out_idx + 1;
                end
            
            % No change - keep the old phase
            else
                
                for j = 1:t_len
                    x(out_idx) = phase_lut(p_acc+1);
                    out_idx = out_idx + 1;
                end
                
            end
            
            % Save the new Q channel state
            q_state = q_bit;
                        
        % Odd bits --> I channel
        else
            
            % Only change phase if the state has changed
            i_bit = bits(i);
            if i_bit ~= i_state

                % When I and Q are in the same state and I is changing, the
                % phase can only increase. Otherwise, if I and Q are not in the
                % same state and I is changing, the phase can only decrease.
                if i_state == q_state
                    s = 1;
                else
                    s = -1;
                end
                
                % Transition the phase, ensuring it wraps in the lookup table
                for j = 1:t_len
                    p_acc = p_acc + s*p_inc;
                    p_acc = mod(p_acc, n_lut);
                    x(out_idx) = phase_lut(p_acc+1);
                    out_idx = out_idx + 1;
                end
            
            % No change - keep the old phase
            else
                
                for j = 1:t_len
                    x(out_idx) = phase_lut(p_acc+1);
                    out_idx = out_idx + 1;
                end
                
            end
            
            % Save the new I channel state
            i_state = i_bit;
            
        end
        
        % Hold the last phase
        for j = 1:t_hold
            x(out_idx) = phase_lut(p_acc+1);
            out_idx = out_idx + 1;
        end
        
    end
    
    % Finish sending the last bit - this holds the last bit value on I or Q
    % while the other channel sends the last bit of information. I've extended
    % the signal to send t_start extra samples such that the output is an
    % integer number of bit periods, as suggested by figure 2 on page 25.
    for i = 1:(t_len + t_start)
        x(out_idx) = phase_lut(p_acc+1);
        out_idx = out_idx + 1;
    end
    
end