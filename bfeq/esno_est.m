% Estimates Es/N0 and noise variance for a set of received symbols compared to a
% known (noise free) reference. 

function [esno, noise_var] = esno_est(x, x_ref)

    % Normalize to unit symbol energy
    x_n = x/mean(abs(x));

    % Remove the mean, leaving only noise, and take the variance
    % Note: we assume the signal is zero mean at this point, so do not estimate
    % the mean in order to compute the variance.
    noise = x_n - x_ref;
    noise_var = sum(abs(noise).^2)/length(x);
    
    % Es/N0, dB - assumes unit symbol energy
    esno = 10*log10(1/noise_var);
        
    % Noise variance for complex process is half of what we estimated (this is
    % the variance as if we generated the real/imag distributions separately).
    % We scale internally since this is the noise variance estimate that matters
    % for computing log-likelihood ratios.
    noise_var = noise_var/2;

end