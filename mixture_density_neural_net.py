"""
Created on Sun May 10 11:27:56 2020
@author: Dr. Ayodeji Babalola
"""
import matlab_funcs as mfun
import numpy as np

#-----------------------------------------------------------------------------
def mdn(nin ,nhidden,ncenter,dim_target,mix_type= None,prior= None,beta= None):
    """
 MDN Creates a Mixture Density Network with specified architecture.

	Description
	NET = MDN(NIN, NHIDDEN, NCENTRES, DIMTARGET) takes the number of
	inputs,  hidden units for a 2-layer feed-forward  network and the
	number of centres and target dimension for the  mixture model whose
	parameters are set from the outputs of the neural network. The fifth
	argument MIXTYPE is used to define the type of mixture model.
	(Currently there is only one type supported: a mixture of Gaussians
	with a single covariance parameter for each component.) For this
	model, the mixture coefficients are computed from a group of softmax
	outputs, the centres are equal to a group of linear outputs, and the
	variances are  obtained by applying the exponential function to a
	third group of outputs.

	The network is initialised by a call to MLP, and the arguments PRIOR,
	and BETA have the same role as for that function. Weight
	initialisation uses the Matlab function RANDN  and so the seed for
	the random weight initialization can be  set using RANDN('STATE', S)
	where S is the seed value. A specialised data structure (rather than
	GMM) is used for the mixture model outputs to improve the efficiency
	of error and gradient calculations in network training. The fields
	are described in MDNFWD where they are set up.

	The fields in NET are
	  
	  type = 'mdn'
	  nin = number of input variables
	  nout = dimension of target space (not number of network outputs)
	  nwts = total number of weights and biases
	  mdnmixes = data structure for mixture model output
	  mlp = data structure for MLP network

	See also
	MDNFWD, MDNERR, MDN2GMM, MDNGRAD, MDNPAK, MDNUNPAK, MLP


	Copyright (c) Ian T Nabney (1996-2001)
	David J Evans (1998)

 Currently ignore type argument: reserved for future use    
    """
    
    net = {}
    net['type'] = 'mdn'

# Set up the mixture model part of the structure
# For efficiency we use a specialised data structure in place of GMM  
    mdnmixes = {}
    mdnmixes['type'] = 'mdnmixes'
    mdnmixes['ncentres'] = ncentres
    mdnmixes['dim_target'] = dim_target
# This calculation depends on spherical variances
    mdnmixes['nparams'] = ncentres + ncentres*dim_target + ncentres
    mdnmixes['mixcoeffs']  = None
    mdnmixes['centres']  = None
    mdnmixes['covars']  = None
    
# Number of output nodes = number of parameters in mixture model
    nout = mdnmixes['nparams']
    if mix_type is not None:
        mlpnet = mlp(nin,nhidden,nout,'linear')
    elif(prior is not None):
        mlpnet = mlp(nin,nhidden,nout,'linear',prior)
    elif(beta is not None):
        mlpnet = mlp(nin,nhidden,nout,'linear',prior,beta)
        
    net['mdnmixes'] = mdnmixes
    net['mlp'] = mlpnet
    net['nin'] = nin
    net.nout = nout
    net.nwts = mlpnet['nwts']
    
    return net

#-----------------------------------------------------------------------------
def mdn2gmm(mdnmixes):
    """
MDN2GMM Converts an MDN mixture data structure to array of GMMs.

	Description
	GMMMIXES = MDN2GMM(MDNMIXES) takes an MDN mixture data structure
	MDNMIXES containing three matrices (for priors, centres and
	variances) where each row represents the corresponding parameter
	values for a different mixture model  and creates an array of GMMs.
	These can then be used with the standard Netlab Gaussian mixture
	model functions.

	See also
	GMM, MDN, MDNFWD


	Copyright (c) Ian T Nabney (1996-2001)
	David J Evans (1998)

 Check argument for consistency

    """
    errstring  = mdnmixes['mdnmixes']
    if errstring is None:
        raise Exception (errstring)
    nmixes = np.shape(mdnmixes['center'])[0]
    # Construct ndata structures containing the mixture model information.
    # First allocate the memory.
    tempmix = gmm(MDNMIXES['dim_target'], mdnmixes['ncentres'],'spherical')
    f = tempmix.key()
    gmmmixes = mfun.cell(len(f),nmixes)
    gmmmixes = cell2struct(gmmmixes, f)  # debug
    
    for i in range(nmixes):
        centers = np.tile(mdnmixes['center'][i,:],
                          (mdnmixes['dim_target'],mdnmixes['ncentres']))
        gmmmixes[i] = gmmunpak(mdnmixes['mixcoeffs'][i,:],
                               centers,mdnmixes['covar'][i,:])
    return gmmmixes

#-----------------------------------------------------------------------------
def gmmunpak(mix,p):
    """
GMMUNPAK Separates a vector of Gaussian mixture model parameters into its components.

	Description
	MIX = GMMUNPAK(MIX, P) takes a GMM data structure MIX and  a single
	row vector of parameters P and returns a mixture data structure
	identical to the input MIX, except that the mixing coefficients
	PRIORS, centres CENTRES and covariances COVARS  (and, for PPCA, the
	lambdas and U (PCA sub-spaces)) are all set to the corresponding
	elements of P.

	See also
	GMM, GMMPAK


	Copyright (c) Ian T Nabney (1996-2001)

    """    
    
    errstring = mix ['gmm']
    
    if errstring is None:
        raise Exception (errstring)
    
    if mix['nwts'] != len(p):
        raise Exception('Invalid weight vector length')
    
    mark1 = mix['centres']
    mark2 = mark1 +  mix['centres']* mix['nin']
    
    mix['prior'] = np.tile(p[0:mark1],(1, mix['centers']))
    mix['centres'] = np.tile(p[ mark1 + 0:mark1],( mix['centers'], mix['nin']))
    
    if mix['covar_type'] == 'spherical':
        mark3 = mix['ncentres'] [2 + mix['nin']]
        mix['covars'] = np.tile(p[mark2 + 0:mark3],(1,mix['ncentres']))
    elif mix['covar_type'] == 'diag':
        mark3 = mix['ncentres'] [1 + mix['nin'+ mix['nin']]
        mix['covars'] = np.tile(p[mark2 + 0:mark3],(mix['ncentres'], mix['nin']))       
    elif mix['covar_type'] == 'full':
        mark3 = mix['ncentres'] [1 + mix['nin'+ mix['nin']**2]
        mix['covars'] = np.tile(p[mark2 + 0:mark3],
                                (mix['nin'],mix['nin'],mix['ncentres']))  
    elif mix['covar_type'] == 'ppca':
        mark3 = mix['ncentres'] [2 + mix['nin']]
        # Now also extract k and eigenspaces
        mark4 = mark3 + mix['ncentres']*mix['ppca_dim']
        mix['lambda'] = np.tile(p[mark3 + 0:mark4],(mix['ncentres'],mix['ppca_dim']))
        mix['U'] = np.tile(p[mark4 + 0:-1],(mix['nin'],mix['ppca_dim'],mix['ncentres']))
    else:
        raise Exception('Unknown covariance type ' + mix['covar_type'])  
    return mix
        
#-----------------------------------------------------------------------------
def gmmpak(mix):  
    """

	Description
	P = GMMPAK(NET) takes a mixture data structure MIX  and combines the
	component parameter matrices into a single row vector P.

	See also
	GMM, GMMUNPAK


	Copyright (c) Ian T Nabney (1996-2001)
    """
    errstring = mix['gmm']
    
    if errstring is None:
        raise Exception ('')
    
    p = [mix['priors'],mix['centres'],mix['covars']]
    
    if mix['covar_type'] == 'ppca':
        p = [p,mix['lambda'],mix['U']]
    return p

      
#-----------------------------------------------------------------------------
def mlp(nin, nhidden, nout, outfunc, prior = None, beta= None):
    """
  MLP	Create a 2-layer feedforward network.

	Description
	NET = MLP(NIN, NHIDDEN, NOUT, FUNC) takes the number of inputs,
	hidden units and output units for a 2-layer feed-forward network,
	together with a string FUNC which specifies the output unit
	activation function, and returns a data structure NET. The weights
	are drawn from a zero mean, unit variance isotropic Gaussian, with
	varianced scaled by the fan-in of the hidden or output units as
	appropriate. This makes use of the Matlab function RANDN and so the
	seed for the random weight initialization can be  set using
	RANDN('STATE', S) where S is the seed value.  The hidden units use
	the TANH activation function.

	The fields in NET are
	  type = 'mlp'
	  nin = number of inputs
	  nhidden = number of hidden units
	  nout = number of outputs
	  nwts = total number of weights and biases
	  actfn = string describing the output unit activation function:
	      'linear'
	      'logistic
	      'softmax'
	  w1 = first-layer weight matrix
	  b1 = first-layer bias vector
	  w2 = second-layer weight matrix
	  b2 = second-layer bias vector
	 Here W1 has dimensions NIN times NHIDDEN, B1 has dimensions 1 times
	NHIDDEN, W2 has dimensions NHIDDEN times NOUT, and B2 has dimensions
	1 times NOUT.

	NET = MLP(NIN, NHIDDEN, NOUT, FUNC, PRIOR), in which PRIOR is a
	scalar, allows the field NET.ALPHA in the data structure NET to be
	set, corresponding to a zero-mean isotropic Gaussian prior with
	inverse variance with value PRIOR. Alternatively, PRIOR can consist
	of a data structure with fields ALPHA and INDEX, allowing individual
	Gaussian priors to be set over groups of weights in the network. Here
	ALPHA is a column vector in which each element corresponds to a
	separate group of weights, which need not be mutually exclusive.  The
	membership of the groups is defined by the matrix INDX in which the
	columns correspond to the elements of ALPHA. Each column has one
	element for each weight in the matrix, in the order defined by the
	function MLPPAK, and each element is 1 or 0 according to whether the
	weight is a member of the corresponding group or not. A utility
	function MLPPRIOR is provided to help in setting up the PRIOR data
	structure.

	NET = MLP(NIN, NHIDDEN, NOUT, FUNC, PRIOR, BETA) also sets the
	additional field NET.BETA in the data structure NET, where beta
	corresponds to the inverse noise variance.

	See also
	MLPPRIOR, MLPPAK, MLPUNPAK, MLPFWD, MLPERR, MLPBKP, MLPGRAD


	Copyright (c) Ian T Nabney (1996-2001
    """
    net  = {}
    net['type'] = 'mlp'
    net['nin'] = nin
    net['nhidden'] = nhidden
    net['nout'] = nout
    net['nwt'] = (nin + 1)*nhidden + (nhidden + 1)*nout
    
    outfuns = ['linear', 'logistic', 'softmax']   
    if outfunc in outfuns:
        net['outfn']  = outfunc
    else:
        raise Exception ('Undefined output function. Exiting.')
        
    if prior is not None:
        if type(prior) == dict:
            net['alpha'] = prior['alpa']
            net['index'] = prior['index']
        elif (len(prior ==1)):
            net['alpha'] = prior
        else:
            raise Exception ('prior must be a scalar or a structure')
    

    net['w1'] = np.random.normal(loc=0,scale=1,size = (nin,nhidden))/ np.sqrt(nin+1)
    net['b1'] = np.random.normal(loc=0,scale=1,size = (1,nhidden))/ np.sqrt(nin+1)
    net['w2'] = np.random.normal(loc=0,scale=1,size = (nin,nhidden))/ np.sqrt(nhidden+1)
    net['b2'] = np.random.normal(loc=0,scale=1,size = (1,nhidden))/ np.sqrt(nhidden+1)    
    
    if beta is not None:
        net['beta'] = beta
    return net

#-----------------------------------------------------------------------------
def mdninit(net, prior = None, t = None, options = None):
    """
 MDNINIT Initialise the weights in a Mixture Density Network.

	Description

	NET = MDNINIT(NET, PRIOR) takes a Mixture Density Network NET and
	sets the weights and biases by sampling from a Gaussian distribution.
	It calls MLPINIT for the MLP component of NET.

	NET = MDNINIT(NET, PRIOR, T, OPTIONS) uses the target data T to
	initialise the biases for the output units after initialising the
	other weights as above.  It calls GMMINIT, with T and OPTIONS as
	arguments, to obtain a model of the unconditional density of T.  The
	biases are then set so that NET will output the values in the
	Gaussian  mixture model.

	See also
	MDN, MLP, MLPINIT, GMMINIT


	Copyright (c) Ian T Nabney (1996-2001)
	David J Evans (1998)

 Initialise network weights from prior: this gives noise around values
 determined later    
    """
    net['mlp'] = mlpinit(net.mlp, prior)
    if prior is not None :
        
        #  Initialise priors, centres and variances from target data
        temp_mix = gmm(net['mdnmixes']['dim_target'],net['mdnmixes']['ncentres'], 'spherical')
        temp_mix = gmminit(temp_mix, t, options)
        ncentres = net['mdnmixes']['ncentres']
        dim_target = net['mdnmixes']['dim_target']
        
        # Now set parameters in MLP to yield the right values.
        # This involves setting the biases correctly.
        
        # Priors
        net['mlp']['b2'][0:ncentres] = temp_mix['priors']
        # Centres are arranged in mlp such that we have
        # u11, u12, u13, ..., u1c, ... , uj1, uj2, uj3, ..., ujc, ..., um1, uM2, 
        # ..., uMc
        # This is achieved by transposing temp_mix.centres before reshaping        
    
        end_centres = ncentres*(dim_target+1);
        net['mlp']['b2'][ncentres+1:end_centres] = np.tile(
            np.transpose(temp_mix ['centres'], 1, ncentres*dim_target)

        net['mlp']['b2'][end_centres+1:net['mlp']['nout']] = np.log(temp_mix['covars'])
        return net
    
#-----------------------------------------------------------------------------
def netopt(net, options, x, t, alg):
    """
    NETOPT	Optimize the weights in a network model. 

	Description

	NETOPT is a helper function which facilitates the training of
	networks using the general purpose optimizers as well as sampling
	from the posterior distribution of parameters using general purpose
	Markov chain Monte Carlo sampling algorithms. It can be used with any
	function that searches in parameter space using error and gradient
	functions.

	[NET, OPTIONS] = NETOPT(NET, OPTIONS, X, T, ALG) takes a network
	data structure NET, together with a vector OPTIONS of parameters
	governing the behaviour of the optimization algorithm, a matrix X of
	input vectors and a matrix T of target vectors, and returns the
	trained network as well as an updated OPTIONS vector. The string ALG
	determines which optimization algorithm (CONJGRAD, QUASINEW, SCG,
	etc.) or Monte Carlo algorithm (such as HMC) will be used.

	[NET, OPTIONS, VARARGOUT] = NETOPT(NET, OPTIONS, X, T, ALG) also
	returns any additional return values from the optimisation algorithm.

	See also
	NETGRAD, BFGS, CONJGRAD, GRADDESC, HMC, SCG


	Copyright (c) Ian T Nabney (1996-2001)
    """
    optstring = [alg, '(''neterr'', w, options, ''netgrad'', net, x, t)']
    # Extract weights from network as single vector
    w = netpak(net)
    
#-----------------------------------------------------------------------------    
def netpak(net):
    """
NETPAK	Combines weights and biases into one weights vector.

	Description
	W = NETPAK(NET) takes a network data structure NET and combines the
	component weight matrices  into a single row vector W. The facility
	to switch between these two representations for the network
	parameters is useful, for example, in training a network by error
	function minimization, since a single vector of parameters can be
	handled by general-purpose optimization routines.  This function also
	takes into account a MASK defined as a field in NET by removing any
	weights that correspond to entries of 0 in the mask.

	See also
	NET, NETUNPAK, NETFWD, NETERR, NETGRAD


	Copyright (c) Ian T Nabney (1996-2001)
    """       
    pakstr = [net['type'],'pak']
    w = feval(pakstr,net)
    
    if net.keys() == 'mask' :
        w = w[logical(net['mask'])]
        
    # need debugging    
    return w

#-----------------------------------------------------------------------------    
def netunpak((net):
"""
NETUNPAK Separates weights vector into weight and bias matrices. 

	Description
	NET = NETUNPAK(NET, W) takes an net network data structure NET and  a
	weight vector W, and returns a network data structure identical to
	the input network, except that the componenet weight matrices have
	all been set to the corresponding elements of W.  If there is  a MASK
	field in the NET data structure, then the weights in W are placed in
	locations corresponding to non-zero entries in the mask (so W should
	have the same length as the number of non-zero entries in the MASK).

	See also
	NETPAK, NETFWD, NETERR, NETGRAD


	Copyright (c) Ian T Nabney (1996-2001)
"""
                 
                
#-----------------------------------------------------------------------------    
def mdnfwd(net,x):
    """
MDNFWD	Forward propagation through Mixture Density Network.

	Description
	 MIXPARAMS = MDNFWD(NET, X) takes a mixture density network data
	structure NET and a matrix X of input vectors, and forward propagates
	the inputs through the network to generate a structure MIXPARAMS
	which contains the parameters of several mixture models.   Each row
	of X represents one input vector and the corresponding row of the
	matrices in MIXPARAMS  represents the parameters of a mixture model
	for the conditional probability of target vectors given the input
	vector.  This is not represented as an array of GMM structures to
	improve the efficiency of MDN training.

	The fields in MIXPARAMS are
	  type = 'mdnmixes'
	  ncentres = number of mixture components
	  dimtarget = dimension of target space
	  mixcoeffs = mixing coefficients
	  centres = means of Gaussians: stored as one row per pattern
	  covars = covariances of Gaussians
	  nparams = number of parameters

	[MIXPARAMS, Y, Z] = MDNFWD(NET, X) also generates a matrix Y of the
	outputs of the MLP and a matrix Z of the hidden unit activations
	where each row corresponds to one pattern.

	[MIXPARAMS, Y, Z, A] = MLPFWD(NET, X) also returns a matrix A  giving
	the summed inputs to each output unit, where each row  corresponds to
	one pattern.

	See also
	MDN, MDN2GMM, MDNERR, MDNGRAD, MLPFWD


	Copyright (c) Ian T Nabney (1996-2001)
	David J Evans (1998)

 Check arguments for consistency

    """ 

    """
     Check arguments for consistency
    errstring = consist(net, 'mdn', x);
    if ~isempty(errstring)
      error(errstring);
    end
    """
    
    # Extract mlp and mixture model descriptors
    mlpnet = net['mlp']
    mixes = net['mdnmixes']
    
    ncentres = mixes['ncentres'] # Number of components in mixture model
    dim_target = mixes['dim_target'] # Dimension of targets
    nparams = mixes['nparams'] # Number of parameters in mixture model
    
    # propagate forward through NLP
    y,z,a = mlpfwd(mlpnet,x)
    # Compute the postion for each parameter in the whole
    # matrix.  Used to define the mixparams structure    
    mixcoeffs = np.linspace(ncentres)
    centers = [ncentres+1:1:[ncentres*[1+dim_target]]]
    variances = [[ncentres*[1+dim_target]+1]:1:nparams]
    
    # Convert output values into mixture model parameters
    
    # Use softmax to calculate priors
    # Prevent overflow and underflow: use same bounds as glmfwd
    # Ensure that sum(exp(y), 2) does not overflow    
    
    maxcut = np.log(realmax) -   np.log(ncentres)
    
    # Ensure that exp(y) > 0
    mincut = log(realmin)
    temp = min(y[:,1:ncentres], maxcut)
    temp = max(temp, mincut)
    temp = np.exp(temp);
    mixpriors = temp./(sum(temp, 2)*np.ones(1,ncentres))
    
    # Centres are just copies of network outputs
    mixcentres =  y[:,[ncentres+1]:ncentres*[1+dim_target]];
    
    # Variances are exp of network outputs
    mixwidths = np.exp(y[:,[ncentres*[1+dim_target+1]:nparams]];
    
    # Now build up all the mixture model weight vectors
    ndata = size(x, 1);
    
    # Return parameters
    mixparams['type'] = mixes['type'] 
    mixparams['ncentres'] = mixes['ncentres']
    mixparams['dim_target'] = mixes['dim_target']
    mixparams['nparams'] = mixes['nparams']
    
    mixparams['mixcoeffs'] = mixpriors;
    mixparams['ncentres']   = mixcentres
    mixparams['covar']     = mixwidths
    
    return mixparams, y, z, a

#-----------------------------------------------------------------------------    
def mlpfwd(net, x):
    """
    MLPFWD	Forward propagation through 2-layer network.

	Description
	Y = MLPFWD(NET, X) takes a network data structure NET together with a
	matrix X of input vectors, and forward propagates the inputs through
	the network to generate a matrix Y of output vectors. Each row of X
	corresponds to one input vector and each row of Y corresponds to one
	output vector.

	[Y, Z] = MLPFWD(NET, X) also generates a matrix Z of the hidden unit
	activations where each row corresponds to one pattern.

	[Y, Z, A] = MLPFWD(NET, X) also returns a matrix A  giving the summed
	inputs to each output unit, where each row corresponds to one
	pattern.

	See also
	MLP, MLPPAK, MLPUNPAK, MLPERR, MLPBKP, MLPGRAD


	Copyright (c) Ian T Nabney (1996-2001)

   Check arguments for consistency
    """  
    eps = 2.2204e-16    
    ndata = np.shape[0]
    z = np.tanh(x*net.w1 + np.ones(ndata, 1)*net['b1'])
    a = z*net['w2'] + np.ones(ndata)*net['b2']
    
    if net['outfun'] == 'linear':
        y=1
    elif net['outfun'] == 'logistic':
        #Prevent overflow and underflow: use same bounds as mlperr
        #Ensure that log(1-y) is computable: need exp(a) > eps
        maxcut = - np.log(eps)
        # Ensure that log(y) is computable
        mincut = - np.log(1/realmin - 1)
    elif net['outfun'] == 'softmax' :
       # Prevent overflow and underflow: use same bounds as glmerr
       # Ensure that sum(exp(a), 2) does not overflow   
        maxcut = np.log(realmax) - np.log(net['nout'])
      # Ensure that exp(a) > 0  
        mincut = np.log(realmin)
        a = min(a, maxcut)
        a = max(a, mincut)
        temp = np.exp(a);
        y = temp./(sum(temp, 2)*np.ones(net['nout']))
    else:
       raise Exception('Unknown activation function' + net['nout']) 
    return y, z, a


    

    
    
    
    


