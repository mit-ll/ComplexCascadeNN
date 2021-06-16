# ComplexCascadeNN (complex split-activation skip-layers feedforward neural network)
Pleaes note the license and disclaimers in this README

# Description

This is a matlab class intended to replace feedforwardnet which takes only real inputs.
When complex inputs are separated into [real; imag], the ability to do complex operations 
typical of signal processing is made more difficult by introducing unnecessary degrees of 
freedom.  complexcascade class directly implements complex neural networks with either complex
activation functions (e.g. tanh) or split real/imag activation functions 
i.e., f(z) =  f(real(z)) + 1i * f(imag(z)).

Most non-linear problems will get stuck in local minima when using gradient techniques, so 
the preferred method is to use a blended Hessian/gradient approach (Levenberg Marquardt).

Bayesian Regularization (MacKay) uses a mahalanobis distance metric that combines the network
error with the netowrk weights norm, adapting the estimate of the variance of each at each 
step.  The approach uses an estimate of the number of parameters in the network problem.

Cascade architecture adds connections from the input directly to deeper 
layers and also connections skipping layers to benefit signal subtraction problems.


# Licenses
see LICENSE file

# Citation
"Application of Complex Split-Activation Feedforward Networks to Beamforming", 
to appear in 55th Asilomar Conference on Signals, Systems, and Computers 2021
Swaroop Appadwedula
MIT Lincoln Laboratory
Lexington, MA, USA
swaroop@ll.mit.edu


# Disclaimer
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the 
Defense Advanced Research Projects Agency under Air Force Contract 
No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations 
expressed in this material are those of the author(s) and do not necessarily 
reflect the views of the Defense Advanced Research Projects Agency.

© 2021 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in 
DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright 
notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 
or DFARS 252.227-7014 as detailed above. Use of this work other than as 
specifically authorized by the U.S. Government may violate any copyrights 
that exist in this work.


# Third-party software License for gradient-descent

MIT License

Copyright (c) 2019 John Malik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
