# ComplexCascadeNN (complex split-activation skip-layers feedforward neural network)
Please note the license and disclaimers in this README

# Description

This is a matlab class intended to replace feedforwardnet which takes only real inputs.
When complex inputs are separated into [real; imag], the ability to do complex operations 
typical of signal processing is made more difficult by introducing unnecessary degrees of 
freedom.  complexcascade class directly implements complex neural networks with either complex
activation functions (e.g. tanh) or split real/imag activation functions 
i.e., f(z) =  f(real(z)) + 1i * f(imag(z)).

Most non-linear problems will get stuck in local minima when using gradient techniques, so 
the preferred method is to use a blended Hessian/gradient approach (Levenberg Marquardt).  
Implementation of LM in this code follows the notation in the companion paper.

Also implemented here is Bayesian Regularization.  Bayesian Regularization (MacKay) uses a 
mahalanobis distance metric that combines the network error with the netowrk weights norm, 
adapting the estimate of the variance of each at each step.  The approach uses an estimate 
of the number of parameters in the network problem.

Skip connections enable feed forwardnets to easily implement signal subtraction.  They 
also help the issue of vanishing gradients when backpropagting errors from the metric.

# Licenses
see LICENSE file

# Citation
"Application of Complex Split-Activation Feedforward Networks to Beamforming", 
to appear in 55th Asilomar Conference on Signals, Systems, and Computers 2021
Swaroop Appadwedula
MIT Lincoln Laboratory
Lexington, MA, USA
swaroop@ll.mit.edu

[![DOI](https://zenodo.org/badge/421960129.svg)](https://zenodo.org/badge/latestdoi/421960129)

# Disclaimer
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Defense Advanced Research Projects Agency under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Defense Advanced Research Projects Agency.

© 2021 Massachusetts Institute of Technology.

    Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
    SPDX-License-Identifier: BSD 3-Clause License

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.


# Third-party software License for gradient-descent

Code can be obtained from
https://github.com/jrvmalik/gradient-descent

MIT License provided at
https://github.com/jrvmalik/gradient-descent/blob/master/LICENSE
