# MLforRF


This is a matlab class intended to replace feedforwardnet which takes only real inputs.
When complex inputs are separated into [real; imag], the ability to do complex operations 
typical of signal processing is made more difficult by introducing unnecessary degrees of 
freedom.  complexnet class directly implements complex neural networks with either complex
activation functions (e.g. tanh) or split real/imag activation functions 
i.e., f(z) =  f(real(z)) + 1i * f(imag(z)).  For example, beamforming or taking the dot 
product is a complex operation (and is covered in the complexnet weight mechanics); 
whereas, making a symbol decision is a split real/imag operation

Most non-linear problems will get stuck in local minima when using gradient techniques, so 
the preferred method is to use a blended Hessian/gradient approach (Levenberg Marquardt)

