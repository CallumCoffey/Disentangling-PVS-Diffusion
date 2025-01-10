import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import csv

# Function to create the tensor
def tensor(axial, radial):
    return np.array([[radial, 0, 0],
                     [0, axial, 0],
                     [0, 0, radial]])

def rotation_matrix_z(angle):
    """
    Generate a 3D rotation matrix for a rotation around the z-axis by the given angle.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

# Function to calculate the error between ADC from model and true ADC
def error(ADC_pvs, D_PVS):
    true_ADC_pvs = apparent_diffusivity(D_PVS)
    # return ADC_pvs
    return (ADC_pvs - true_ADC_pvs) / true_ADC_pvs * 100

def forward_model(pvs_fraction, D_wm, D_PVS, bvals, bvecs, SNR='inf'):
    """
    Compute the signal attenuation for each gradient direction and b-value
    using the Stejkal-Tanner equation for two Tensors: WM and PVS.
    Optionally add Gaussian noise based on the specified SNR.
    
    Args:
        pvs_fraction (float): Fractional volume of the perivascular space Tensor.
        D_wm (numpy.ndarray): Diffusion tensor for the white matter Tensor (3x3 matrix).
        D_PVS (numpy.ndarray): Diffusion tensor for the perivascular space Tensor (3x3 matrix).
        bvals (numpy.ndarray): Array of b-values.
        bvecs (numpy.ndarray): Array of gradient direction vectors (each vector is 3 elements).
        SNR (float or str): Signal-to-noise ratio (in dB). If 'inf', no noise is added.
    
    Returns:
        numpy.ndarray: Array of computed signal attenuations, with optional noise.
    """
    # Initialize an empty list to store the signals for each b-value and gradient direction
    signals = []

    wm_fraction = 1-pvs_fraction
    
    # Iterate over b-values and corresponding gradient directions
    for b, g in zip(bvals, bvecs):
    
        # Compute signal attenuation for both Tensors (WM and PVS)
        signal_wm = wm_fraction * np.exp(-b * (g.T @ D_wm @ g).item())
        signal_pvs = pvs_fraction * np.exp(-b * (g.T @ D_PVS @ g).item())
        
        # The total signal is the sum of the signals from both Tensors
        total_signal = signal_wm + signal_pvs

        # Add Gaussian noise if SNR is not 'inf'
        if SNR != 'inf':
            # Calculate noise based on SNR (signal-to-noise ratio)
            # Generate Gaussian noise
            noise = np.random.normal(0, total_signal / SNR)

            # Add noise to the signal
            total_signal += noise
        
        # Append the calculated (or noisy) signal to the signals list
        signals.append(total_signal)
    
    # Convert the list of signal attenuations to a numpy array
    return np.array(signals)

def backward_model(signals, bvals, bvecs):
    # Number of gradient directions (assumes len(bvals) == len(bvecs) == len(signals))
    num_gradients = len(signals)
    
    # Initialize the matrix A and vector b for least-squares fitting
    A = np.zeros((num_gradients, 6))  # 6 parameters for the diffusion tensor
    b = np.zeros(num_gradients)
    
    for i in range(num_gradients):
        # Gradient direction and b-value for the current measurement
        bval = bvals[i]
        gvec = bvecs[i]
        
        # Form the matrix A
        A[i, 0] = gvec[0] * gvec[0]  # Dxx
        A[i, 1] = gvec[0] * gvec[1] * 2  # Dxy
        A[i, 2] = gvec[0] * gvec[2] * 2  # Dxz
        A[i, 3] = gvec[1] * gvec[1]  # Dyy
        A[i, 4] = gvec[1] * gvec[2] * 2  # Dyz
        A[i, 5] = gvec[2] * gvec[2]  # Dzz

        # Form the vector b for negative log of the signal
        if signals[i] > 0:
            # print(signals[i])
            b[i] = -np.log(signals[i]) / bval           
        else:
            b[i] = 0

    # Solve the system of equations using least squares
    tensor_params = np.linalg.lstsq(A, b, rcond=None)[0]

    # Reconstruct the diffusion tensor from the parameters
    D = np.array([[tensor_params[0], tensor_params[1], tensor_params[2]],
                [tensor_params[1], tensor_params[3], tensor_params[4]],
                [tensor_params[2], tensor_params[4], tensor_params[5]]])

    return D

def backward_model_multi_Tensor(signals, bvals, bvecs, PVS_constrain=False):
    """
    Estimate parameters for a two-Tensor model with white matter (WM) and PVS Tensors.
    Optionally constrain the PVS diffusion tensor to only diagonal elements, and reduce the number
    of parameters by assuming the second and third eigenvalues of the PVS tensor are equal.

    Args:
        signals (np.ndarray): Measured diffusion-weighted signal (1D array).
        bvals (np.ndarray): b-values (1D array corresponding to signal measurements).
        bvecs (np.ndarray): Gradient directions (Nx3 array).
        PVS_constrain (bool): If True, constrain the PVS tensor to only diagonal elements.

    Returns:
        dict: Estimated parameters (f_WM, f_PVS, D_WM, D_PVS).
    """
    # Define the model function
    def model_function(params, bvals, bvecs):
        f_WM = params[0]
        f_PVS = 1 - f_WM
        
        D_WM_flat = params[1:7]
        if PVS_constrain:
            # Use only two parameters for the PVS tensor: one for the first eigenvalue,
            # and the second and third eigenvalues are assumed equal
            D_PVS_flat = params[7:9]  # Only two parameters for PVS tensor
            # Create a diagonal tensor where the second and third eigenvalues are equal
            D_PVS = np.diag([D_PVS_flat[1], D_PVS_flat[0], D_PVS_flat[1]])
        else:
            D_PVS_flat = params[7:13]  # Full tensor elements
            D_PVS = np.array([[D_PVS_flat[0], D_PVS_flat[1], D_PVS_flat[2]],
                              [D_PVS_flat[1], D_PVS_flat[3], D_PVS_flat[4]],
                              [D_PVS_flat[2], D_PVS_flat[4], D_PVS_flat[5]]])

        # Reconstruct WM tensor
        D_WM = np.array([[D_WM_flat[0], D_WM_flat[1], D_WM_flat[2]],
                         [D_WM_flat[1], D_WM_flat[3], D_WM_flat[4]],
                         [D_WM_flat[2], D_WM_flat[4], D_WM_flat[5]]])

        # Check for negative eigenvalues
        def check_positive_semi_definite(tensor):
            eigvals = np.linalg.eigvals(tensor)
            return np.all(eigvals >= 0)

        # Penalize if the tensor is not positive semi-definite
        penalty = 0
        if not check_positive_semi_definite(D_WM):
            penalty += 1e6  # Add large penalty for negative eigenvalues in D_WM
        if not check_positive_semi_definite(D_PVS):
            penalty += 1e6  # Add large penalty for negative eigenvalues in D_PVS

        # Calculate signal contributions for each Tensor
        signal_WM = np.exp(-bvals * np.einsum('ij,ij->i', bvecs @ D_WM, bvecs))
        signal_PVS = np.exp(-bvals * np.einsum('ij,ij->i', bvecs @ D_PVS, bvecs))
        
        # Combine signals with volume fractions
        signal_model = f_WM * signal_WM + f_PVS * signal_PVS
        return signal_model, penalty

    # Define the cost function (sum of squared errors with penalty for non-positive semi-definite tensors)
    def cost_function(params):
        predicted_signal, penalty = model_function(params, bvals, bvecs)
        cost = np.sum((signals - predicted_signal) ** 2) + penalty  # Add penalty to cost
        return cost

    # Initial guesses for parameters
    initial_guess = [0.5,  # Initial guess for f_WM
                     1e-3, 0, 0, 1e-3, 0, 1e-3]  # Initial guess for flattened D_WM

    if PVS_constrain:
        # Initial guess for the PVS tensor (only two parameters)
        initial_guess += [2e-3, 1e-3]  # First eigenvalue and the shared second/third eigenvalue
        bounds = [(0, 1)] + [(-2e-3, 2e-3)] * 6 + [(-3e-3, 3e-3)] * 2  # Diagonal bounds for PVS
    else:
        initial_guess += [2e-3, 0, 0, 2e-3, 0, 2e-3]  # Flattened full D_PVS
        bounds = [(0, 1)] + [(-2e-3, 2e-3)] * 6 + [(-3e-3, 3e-3)] * 6  # Full tensor bounds

    # Perform optimization
    result = minimize(cost_function, initial_guess, bounds=bounds, method='L-BFGS-B')

    # Extract optimized parameters
    f_WM_opt = result.x[0]
    f_PVS_opt = 1 - f_WM_opt
    D_WM_flat_opt = result.x[1:7]
    D_WM_opt = np.array([[D_WM_flat_opt[0], D_WM_flat_opt[1], D_WM_flat_opt[2]],
                         [D_WM_flat_opt[1], D_WM_flat_opt[3], D_WM_flat_opt[4]],
                         [D_WM_flat_opt[2], D_WM_flat_opt[4], D_WM_flat_opt[5]]])

    if PVS_constrain:
        D_PVS_flat_opt = result.x[7:9]
        D_PVS_opt = np.diag([D_PVS_flat_opt[1], D_PVS_flat_opt[0], D_PVS_flat_opt[1]])
    else:
        D_PVS_flat_opt = result.x[7:13]
        D_PVS_opt = np.array([[D_PVS_flat_opt[0], D_PVS_flat_opt[1], D_PVS_flat_opt[2]],
                              [D_PVS_flat_opt[1], D_PVS_flat_opt[3], D_PVS_flat_opt[4]],
                              [D_PVS_flat_opt[2], D_PVS_flat_opt[4], D_PVS_flat_opt[5]]])

    print('Fraction =', f_WM_opt, f_PVS_opt)
    print('WM Tensor')
    print(D_WM_opt)
    print('PVS tensor')
    print(D_PVS_opt)
    return f_WM_opt, f_PVS_opt, D_WM_opt, D_PVS_opt

def apparent_diffusivity(D_tensor):
    """
    Calculate the apparent diffusion coefficient (ADC) along the y-axis of the diffusion tensor.
    
    Args:
        D_tensor (numpy.ndarray): The diffusion tensor (3x3 matrix).
        
    Returns:
        float: The apparent diffusivity along the y-axis.
    """
    # Unit vector along the y-axis
    e_y = np.array([0, 1, 0])
    
    # Calculate the apparent diffusivity along the y-axis using the projection of the tensor
    adc_y = e_y.T @ D_tensor @ e_y
    
    return adc_y

def modelDiffusion(D_PVS_axial, D_PVS_radial, alpha, f_PVS, SNR = 'inf'):
    # Create tensors for PVS and WM
    D_PVS = tensor(D_PVS_axial, D_PVS_radial)
    
    # Apply rotation to the white matter tensor
    R = rotation_matrix_z(alpha)
    D_wm_rotated = R @ D_wm @ R.T
    
    # Compute the signal with SNR
    np.random.seed(np.random.randint(0,100000))
    signals = forward_model(f_PVS, D_wm_rotated, D_PVS, bvals, bvecs, SNR)
    # D_combined = backward_model(signals, bvals, bvecs)
    # _, _, _, D_combined = backward_model_multi_Tensor(signals, bvals, bvecs)
    _, _, _, D_combined = backward_model_multi_Tensor(signals, bvals, bvecs,True)
    
    # Calculate ADC for this combination of parameters
    ADC_PVS = apparent_diffusivity(D_combined)
    
    # Store the result    
    return(error(ADC_PVS, D_PVS))

# Default parameters
f_PVS_default = 0.2                              # Volume fraction of PVS
alpha_default = np.deg2rad(45)                   # Default rotation angle of white matter

D_wm_axial = 2e-3                                # Default axial diffusivity for white matter
D_wm_radial = 5e-4                               # Default radial diffusivity for white matter
D_wm = tensor(D_wm_axial, D_wm_radial)           # Default tensor for white matter

D_PVS_axial_default = 2.5e-3                     # Default axial diffusivity for PVS
D_PVS_radial_default = 1e-3                    # Default radial diffusivity for PVS
D_PVS_default = tensor(D_PVS_axial_default, D_PVS_radial_default)   # Default tensor for PVS

data = """
-0.0257519901789885   -0.918531649763922    0.394507849584504                   65
  -0.612633586119227    0.049099966831541    0.788840467024757                  985
  -0.363254007316178    0.874727017538958   -0.320779006414287                 1985
  -0.713543217415011   -0.482787147110156   -0.507713154710681                  990
 -0.0765589811801163   -0.376607907400572   -0.923203773001402                 2025
  -0.165159904191849     0.93692745635376   -0.308040821284797                 1010
  -0.931644026344895   -0.305564008614725    0.196647005609476                 1990
  -0.912469789048039     0.38773591037792   -0.130612969792562                 1005
  -0.270749849293202   -0.466829740088279    0.841881531378862                 2025
   -0.20295492689668    0.176127936597119   -0.963217653284243                  975
   -0.99866832962721   0.0449490148412247  -0.0253210083606899                   60
   -0.34510398651612   -0.883668965441276   -0.316278987614773                 2000
  -0.117480897300292    0.830889273602065    0.543894524501352                  990
  -0.792847439710411    0.520099632506829    0.317630775604171                 1990
  -0.674207927917981   -0.653668930117434    0.343744963209168                 1010
  -0.773977214709677    0.085422023691068   -0.627425174007844                 1975
   -0.60321177570307    0.477557822502431   -0.638806762503252                 1015
  -0.534818654408496    0.193737874803078    0.822456468613065                 2020
  -0.674416738213931    0.584375773212071    0.451294824809322                  990
  -0.376324891321189   -0.870208748748998    0.317987908217905                 2010
 -0.0326729842202653   -0.371470820603017   -0.927869552007535                   55
  -0.398545986612698   -0.907252969528906   -0.134361995504281                 1000
  -0.801713122717734   -0.425242065109406   -0.420030064309291                 2010
  -0.981916188939326   -0.159516030706389    0.101957019604083                 1000
  -0.298491703596951    0.880690125591003    0.367814634796243                 2005
  -0.200818950104608   -0.597024851613701    0.776680806917824                  980
  -0.339292011679045    0.412469014174526   -0.845429029047786                 1970
  -0.206509888506653   -0.405319781213058   -0.890544519328691                 1020
  -0.960284178999406    0.253987047299843   -0.115520021499929                 1990
  -0.546637723130867    0.837279575947279   0.0122519937906918                  995
  -0.548336450008438   -0.303514695604671    0.779234218411992                   65
  -0.702286510462872    -0.29488579448441    0.647947548365745                 2015
  -0.837725454622153  -0.0319480173408449   -0.545156295814416                  985
   -0.40659192198863   -0.572754890183984   -0.711782863480096                 2015
  -0.255079789199375    0.277178770999321     0.92634023449773                 1020
  -0.702975382781324    0.615912459283637   -0.355636687790552                 2010
   -0.80082999120021   -0.268678997000071    0.535240994100141                  985
 -0.0303089833993598  -0.0631539654186661     0.99754345377893                 1965
  -0.248406862994405   -0.895802505779825    0.368553796691699                  990
 0.00268799930894672   -0.999031743280197   0.0439129887191295                 2005
  -0.509584493985568    0.825296180476626    0.243330758393108                   60
  -0.341616979514752   -0.752883954932512   -0.562550966324293                 1010
   -0.73244620604777    -0.68016719134436   0.0299190084219513                 1995
  -0.248760906700917    0.656858753602422   -0.711796733002625                 1015
  -0.354302861784269    0.616553759472625    0.703086725668783                 1980
  -0.913235168928833    0.245523045407752    0.325146060110266                  995
  0.0233369975312766   -0.758827919841512    0.650872931235606                 2020
  -0.829895655859303    -0.54305442917337   -0.127926101093727                 1000
  -0.907155798375608    0.105954976397151     0.40723690948905                 2010
  -0.526227672186877  -0.0690879569582771   -0.847532472078865                  980
  -0.678354215566523    0.185072058790867    -0.71104422596491                   55
  -0.454515898502673   -0.116115974100683   -0.883137802705194                 2025
  -0.137497900495729    0.975328293969705    0.172710874994635                 1000
  -0.616263258600676    0.785393055100862   0.0581149300800638                 2000
  -0.275246763806889   -0.199951828405004    0.940350192923535                 1020
   -0.96978833143928   -0.141328048305724   -0.198839068008054                 1995
  -0.676781505497569    0.675864506097572   -0.291845786698952                  990
 -0.0551949634320634   -0.796490472229776    -0.60212660102251                 1985
  -0.412042579817576    0.562277426623985    0.716983268930584                  985
  -0.546480821317542   -0.627397794820139    0.554734818617807                 2015
  -0.609174164327052   -0.756279204033585   -0.238639064410597                   60
   -0.97644492929368 -0.00814499940994728   -0.215612984398604                  995
  -0.632666101244578    0.405879064928598   -0.659542105546471                 2020
  -0.523891811792084   -0.506689817892344    0.684691753989655                 1015
  -0.406968734005069  -0.0929419392611576    0.908701406111318                 1970
   0.141405926899209    0.988362489194468  -0.0560709710196862                 1005
  -0.625837820162609   -0.713228795057388   -0.315644909281142                 1990
  0.0603509880678781    0.121460975995729    0.990759804165165                 1020
  -0.658313205698064    0.438257471198711      0.6120082614982                 2015
  -0.791599305819239    0.592067480814389    0.151084867503672                 1000
 -0.0741109934217783    0.470877958211299   -0.879079921921093                   65
 -0.0107709970800114    -0.92244975000098     0.38596689540041                 2015
    -0.5064066893057      -0.444196727405   -0.739081546508319                 1010
  -0.677581203706333   -0.267422080402499   -0.685105206006403                 2020
  -0.640633163212159   -0.765688195014533   0.0575390146610921                 1000
 -0.0434449987203817    0.405740988003565   -0.912954973008021                 2025
  -0.467823092277097    0.303819059885126   -0.829961163659368                  980
  -0.848877721388472    0.528121826692828  -0.0222249927096982                 1990
   -0.75373943431376    0.247808814004524    0.608660543211111                  985
 -0.0166369890396413   -0.959959367679305   -0.279644815793971                 2000
  -0.694857675164094    0.328472846483026    0.639748700966941                   65
  -0.410278652813777    0.836364292328084    0.363546692412207                 1005
   -0.77915393897847   -0.512913959785827    0.360330971790043                 2010
-0.00710899695495198   -0.804780655294564    0.593529745695991                  985
  -0.221490864000482      0.3676827743008    0.903189445601966                 1975
  -0.836553859000466    0.288066951400161    -0.46604192140026                 1010
  -0.401724871699285    0.683583781598783   -0.609368805398915                 1975
0.000372999801612166    0.653696652221322    0.756756597424683                  985
  -0.994425360452162   0.0207550075190016    0.103380037495027                 1995
  -0.883590087488869   -0.406865040294875     0.23179602299708                  990
  -0.087004993441992   -0.916716930920988   -0.389948970608928                   60
  -0.431384774517057   -0.901924528535662   0.0209599890408288                 1990
  -0.021185994871012   -0.375399909117932    0.926620775644262                  975
   -0.28366683849538   0.0584239667290486   -0.957141454984413                 1965
  -0.895659504075827   -0.311163175091602   -0.317760178791424                 1005
  -0.350705793590232     0.93639944887392    0.012708992519646                 1995
  -0.158630910800304   -0.929489477501781   -0.332994812800638                  995
  -0.751309442487855   0.0169839873997255    0.659731510389336                 2015
  -0.423430836497474    0.793654693495266   -0.436827831297394                 1010
  -0.906419516328654    0.156355089104943   -0.392373223512404                 1985
  -0.706589527609411    0.620067585408258   -0.340950772104541                   60
  -0.550847843410548   -0.227993935204366    0.802860771815373                 1015
  -0.122388938102952   -0.612462690514772   -0.780967605318836                 2015
  -0.984613399575826    0.162230901096017   0.0649429603984055                 1000
  -0.281741954203532   -0.720916882709037    0.633166897007937                 1975
  -0.577187999395785    -0.73268999919465   -0.360581999597367                 1005
  -0.536320533592596    0.719806373990063    0.440725616693915                 2005
  -0.437773078394769   -0.743167133091119    0.506021090593953                  985
  -0.893443556549578   -0.439466273724386  -0.0928870578651543                 1995
  -0.508677627406003    0.338034752403989    0.791820420009344                  980
  -0.788573310198286   -0.573141225398755    0.222848087699516                   60
  -0.558658029893922    0.151155008098355   -0.815508043591127                 1970
  -0.686600438216871    0.128118081803148   -0.715657456717585                  980
  -0.495956767136637   -0.327683846124207    0.804145622459404                 2020
  -0.306069773809805    0.947871299430366  -0.0886639344728404                 1005
  -0.426492078905079   -0.748667138508916   -0.507545093906044                 1985
   -0.90033153659479  -0.0392209798097731    0.433433776897492                 1005
  -0.930081805385145    0.318586933294912    0.182893961697079                 1995
   -0.31416487170773   -0.183367925104512    -0.93149161962292                  980
  -0.577904700906892    0.787446592509391   -0.214368889102557                 1985
  -0.640512336599589   -0.337276177299784   -0.689919362599557                   65
  -0.423445160716613   -0.883488335334663    0.200356076007861                 1005
  -0.211013910896022    0.124034947597662     0.96958159078172                 1970
  -0.363517628203024    0.731280252106084    0.577134409704802                 1005
  -0.861240737219178   -0.170962146303807   -0.478577409710657                 2010
  -0.196542798798664   0.0277669715698113    0.980101996493339                 1020
  -0.601152347284481   -0.755971436780485    0.259081149693312                 1985
  -0.778102836324443    0.615799870419344   -0.123880973903891                 1005
  -0.199932835087183    0.791194347249279    0.577960523162948                 1985
  -0.725138357283371   -0.249139122794287   -0.641953316285279                 1010
  -0.240206780901441   0.0147209865700883    0.970610114805822                   65
  -0.177036880703845    0.614195586213338     -0.7690394819167                 1970
  -0.217583922012595    0.478755828327714   -0.850558695049237                  980
  -0.875875661918972   -0.177932931303854    0.448532826909715                 2010
  -0.254223958296993   -0.656646892192232     -0.7100598833916                  985
  -0.364828043799903   -0.362896043599904   -0.857442102999773                 1970
  -0.945682656706576   -0.322951224302246   -0.037240025860259                 1000
  -0.196714866304451   -0.966340343121866    0.165799887303752                 2010
  -0.610109534193666    0.755521423092156    0.238649817797522                 1000
  -0.823579819478688    0.410429909989379   -0.391488914189869                 1985
  -0.343272956511199    0.913533884129804   -0.218218972307119                   60
  -0.717517921705704   -0.478194947803801    0.506455944704026                  985
  -0.467037568881064    0.431174601982518    0.771987287368699                 2020
""" 
lines = data.strip().split("\n")
rows = [line.split() for line in data.strip().split("\n")]
bvecs = np.array([[float(row[0]), float(row[1]), float(row[2])] for row in rows])
bvals = np.array([float(row[3]) for row in rows])

# Quick visualisation of default tensor settings to ensure reasonable reproduction of base tensors and good estimation of combination
def visualiseTensors(print=False):
    # Function to plot diffusion tensor as ellipsoid in 3D
    def plot_diffusion_tensor_ellipsoid(tensor, ax, label, global_limits):
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(tensor)
        
        # Sort eigenvalues and eigenvectors in descending order of eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Create a unit sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Scale the sphere with the eigenvalues
        x = eigenvalues[0] * x
        y = eigenvalues[1] * y
        z = eigenvalues[2] * z
        
        # Apply the eigenvectors (rotation)
        ellipsoid_points = np.dot(np.vstack([x.ravel(), y.ravel(), z.ravel()]).T, eigenvectors.T)
        
        # Reshape the points back into 3D space
        x, y, z = ellipsoid_points[:, 0].reshape(x.shape), ellipsoid_points[:, 1].reshape(y.shape), ellipsoid_points[:, 2].reshape(z.shape)
        
        # Plot the ellipsoid
        ax.plot_surface(x, y, z, color='b', alpha=0.3)
        ax.set_title(f'{label} Diffusion Tensor')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set the global axis limits for all subplots
        ax.set_xlim(global_limits)
        ax.set_ylim(global_limits)
        ax.set_zlim(global_limits)

        # Hide the z-axis ticks and label
        ax.set_zticks([])

        # Set the view to look at the x and y plane (i.e., top-down view)
        ax.view_init(elev=90, azim=0)

    # Calculate global axis limits by finding min/max values across all tensors
    def get_global_limits(*tensors):
        min_vals = np.inf * np.ones(3)
        max_vals = -np.inf * np.ones(3)
        
        for tensor in tensors:
            eigenvalues, eigenvectors = np.linalg.eig(tensor)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            x = eigenvalues[0] * x
            y = eigenvalues[1] * y
            z = eigenvalues[2] * z
            ellipsoid_points = np.dot(np.vstack([x.ravel(), y.ravel(), z.ravel()]).T, eigenvectors.T)
            x, y, z = ellipsoid_points[:, 0], ellipsoid_points[:, 1], ellipsoid_points[:, 2]
            
            min_vals = np.minimum(min_vals, [x.min(), y.min(), z.min()])
            max_vals = np.maximum(max_vals, [x.max(), y.max(), z.max()])
        
        return min_vals, max_vals

    # Create a figure with a 3x5 grid (3 rows, 5 columns)
    fig = plt.figure(figsize=(15, 12))

    # Assuming D_wm and D_PVS are predefined and true tensors, similar to the original function
    # In practice, these would be input parameters or pre-defined tensors
    R = rotation_matrix_z(alpha_default)
    D_wm_rotated = R @ D_wm @ R.T

    # Compute signals for the combined model
    signals = forward_model(f_PVS_default, D_wm_rotated, D_PVS_default, bvals, bvecs)

    # Estimate the combined tensor using the simple model
    D_combined = backward_model(signals, bvals, bvecs)

    # Estimate the WM and PVS tensors using the multi-Tensor model
    VF_WM, VF_PVS, D_WM_est_1, D_PVS_est_1 = backward_model_multi_Tensor(signals, bvals, bvecs)
    VF_WM, VF_PVS, D_WM_est_2, D_PVS_est_2 = backward_model_multi_Tensor(signals, bvals, bvecs, True)

    # Calculate global axis limits by finding min/max values across all tensors
    min_vals, max_vals = get_global_limits(D_PVS_default, D_wm, D_combined, D_WM_est_1, D_PVS_est_1,
                                            D_WM_est_2, D_PVS_est_2)
    global_limits = [min(min_vals), max(max_vals)]  # This ensures that all axes use the same range

    # Add subplots for each diffusion tensor (arranged in 3 rows, 5 columns)
    rows = 4
    cols = 3
    ax1  = fig.add_subplot(rows, cols, 1, projection='3d')
    ax2  = fig.add_subplot(rows, cols, 2, projection='3d')
    ax3  = fig.add_subplot(rows, cols, 3, projection='3d')

    ax4  = fig.add_subplot(rows, cols, 4, projection='3d')
    ax5  = fig.add_subplot(rows, cols, 5, projection='3d')

    ax7  = fig.add_subplot(rows, cols, 7, projection='3d')
    ax8  = fig.add_subplot(rows, cols, 8, projection='3d')

    # Plot the ellipsoids for the first row (true tensors and combined tensor)
    plot_diffusion_tensor_ellipsoid(D_wm_rotated, ax1, 'True WM', global_limits)
    plot_diffusion_tensor_ellipsoid(D_PVS_default, ax2, 'True PVS', global_limits)
    plot_diffusion_tensor_ellipsoid(D_combined, ax3, 'Combined Estimate', global_limits)

    # Plot the ellipsoids for the second row (estimated tensors for the first model)
    plot_diffusion_tensor_ellipsoid(D_WM_est_1, ax4, 'Estimated WM (1st Model)', global_limits)
    plot_diffusion_tensor_ellipsoid(D_PVS_est_1, ax5, 'Estimated PVS (1st Model)', global_limits)

    # Plot the ellipsoids for the third row (estimated tensors for the second model)
    plot_diffusion_tensor_ellipsoid(D_WM_est_2, ax7, 'Estimated WM (2nd Model)', global_limits)
    plot_diffusion_tensor_ellipsoid(D_PVS_est_2, ax8, 'Estimated PVS (2nd Model)', global_limits)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig('VisualiseTensor_3x5_Grid', dpi=300, bbox_inches='tight')

    if print:
        print('D_PVS = \n', D_PVS_default)
        print('D_WM = \n', D_wm)
        print('D_combined = \n', D_combined)
        print('D_WM_est_1 = \n', D_WM_est_1)
        print('D_PVS_est_1 = \n', D_PVS_est_1)
        print('D_WM_est_2 = \n', D_WM_est_2)
        print('D_PVS_est_2 = \n', D_PVS_est_2)
        
def genSNR(SNR_values, alpha_range, f_PVS_range, D_PVS_axial_range, D_PVS_radial_range, csv_filename="SNR_comparison.csv"):
    """
    Generate error data for different SNR values and save to CSV.
    """
    # Prepare storage for results
    data = []

    for SNR in SNR_values:
        # 1. ADC vs. Alpha
        for alpha in alpha_range:
            errors = []
            if SNR == "inf":
                errors.append(modelDiffusion(D_PVS_axial_default, D_PVS_radial_default, alpha, f_PVS_default, SNR))
            else:
                errors = [modelDiffusion(D_PVS_axial_default, D_PVS_radial_default, alpha, f_PVS_default, SNR) for _ in range(10)]
            data.append(["alpha", SNR, alpha, np.mean(errors)])

        # 2. ADC vs. PVS Axial Diffusivity
        for D_PVS_axial in D_PVS_axial_range:
            errors = []
            if SNR == "inf":
                errors.append(modelDiffusion(D_PVS_axial, D_PVS_radial_default, alpha_default, f_PVS_default, SNR))
            else:
                errors = [modelDiffusion(D_PVS_axial, D_PVS_radial_default, alpha_default, f_PVS_default, SNR) for _ in range(10)]
            data.append(["pvs_axial", SNR, D_PVS_axial, np.mean(errors)])

        # 3. ADC vs. PVS Radial Diffusivity
        for D_PVS_radial in D_PVS_radial_range:
            errors = []
            if SNR == "inf":
                errors.append(modelDiffusion(D_PVS_axial_default, D_PVS_radial, alpha_default, f_PVS_default, SNR))
            else:
                errors = [modelDiffusion(D_PVS_axial_default, D_PVS_radial, alpha_default, f_PVS_default, SNR) for _ in range(10)]
            data.append(["pvs_radial", SNR, D_PVS_radial, np.mean(errors)])

        # 4. ADC vs. PVS Volume Fraction
        for f_PVS in f_PVS_range:
            errors = []
            if SNR == "inf":
                errors.append(modelDiffusion(D_PVS_axial_default, D_PVS_radial_default, alpha_default, f_PVS, SNR))
            else:
                errors = [modelDiffusion(D_PVS_axial_default, D_PVS_radial_default, alpha_default, f_PVS, SNR) for _ in range(10)]
            data.append(["pvs_fraction", SNR, f_PVS, np.mean(errors)])

    # Save data to CSV
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Parameter", "SNR", "Value", "Error"])
        writer.writerows(data)

    print(f"Data successfully saved to {csv_filename}.")


# Function to visualize 3D plot comparing PVS axial and radial diffusivity with error
def genDiffusivities(D_PVS_axial_range, D_PVS_radial_range, csv_filename="PVS_diffusivity_errors_all.csv"):
    """
    Generate errors for different PVS axial and radial diffusivity values and save to CSV.
    """
    # Apply rotation to the WM tensor
    R = rotation_matrix_z(alpha_default)
    D_wm_rotated = R @ D_wm @ R.T

    # Prepare data list for CSV
    data = []

    # Iterate over axial and radial diffusivity ranges
    for D_PVS_axial in D_PVS_axial_range:
        for D_PVS_radial in D_PVS_radial_range:
            if D_PVS_axial > D_PVS_radial:  # Constraint: axial > radial
                D_PVS = tensor(D_PVS_axial, D_PVS_radial)

                # Simulate signals and compute ADC
                signals = forward_model(f_PVS_default, D_wm_rotated, D_PVS, bvals, bvecs)

                # Calculate combined diffusion for each model
                D_combined1 = backward_model(signals, bvals, bvecs)
                _, _, _, D_combined2 = backward_model_multi_Tensor(signals, bvals, bvecs)
                _, _, _, D_combined3 = backward_model_multi_Tensor(signals, bvals, bvecs, True)

                # Compute ADC
                ADC_pvs1 = apparent_diffusivity(D_combined1)
                ADC_pvs2 = apparent_diffusivity(D_combined2)
                ADC_pvs3 = apparent_diffusivity(D_combined3)

                # Calculate errors
                err1 = error(ADC_pvs1, D_PVS)
                err2 = error(ADC_pvs2, D_PVS)
                err3 = error(ADC_pvs3, D_PVS)

                # Store data
                data.append([D_PVS_axial, D_PVS_radial, err1, err2, err3])

    # Save data to CSV
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Axial_Diffusivity", "Radial_Diffusivity", "Error1", "Error2", "Error3"])
        writer.writerows(data)

    print(f"Data successfully saved to {csv_filename}.")


# Function to visualize 3D plot comparing alpha and volume fraction (f_PVS) with error
def genAlphaVolume(alpha_range, f_PVS_range, csv_filename="PVS_alpha_volume_errors.csv"):
    """
    Generate errors for different alpha and f_PVS values and save to CSV.
    """
    # Prepare data list for CSV
    data = []

    # Iterate over alpha and f_PVS ranges
    for alpha in alpha_range:
        # Apply rotation to the WM tensor for each alpha
        R = rotation_matrix_z(alpha)
        D_wm_rotated = R @ D_wm @ R.T

        for f_PVS in f_PVS_range:
            # Simulate signals and compute ADC
            signals = forward_model(f_PVS, D_wm_rotated, D_PVS_default, bvals, bvecs)

            # Calculate combined diffusion for each model
            D_combined1 = backward_model(signals, bvals, bvecs)
            _, _, _, D_combined2 = backward_model_multi_Tensor(signals, bvals, bvecs)
            _, _, _, D_combined3 = backward_model_multi_Tensor(signals, bvals, bvecs, True)

            # Compute ADC
            ADC_pvs1 = apparent_diffusivity(D_combined1)
            ADC_pvs2 = apparent_diffusivity(D_combined2)
            ADC_pvs3 = apparent_diffusivity(D_combined3)

            # Calculate errors
            err1 = error(ADC_pvs1, D_PVS_default)
            err2 = error(ADC_pvs2, D_PVS_default)
            err3 = error(ADC_pvs3, D_PVS_default)

            # Store data
            data.append([alpha, f_PVS, err1, err2, err3])

    # Save data to CSV
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Alpha", "Volume_Fraction", "Error1", "Error2", "Error3"])
        writer.writerows(data)

    print(f"Data successfully saved to {csv_filename}.")

genSNR(
    SNR_values = ['inf', 50, 20],
    alpha_range = np.deg2rad(np.linspace(0, 90, 100)),       # White matter angles (0° to 90°)
    f_PVS_range = np.linspace(0, 1, 100),                # Volume fraction (0 to 1) 
    D_PVS_axial_range = np.linspace(5e-4, 3e-3, 100),      # Axial diffusivity (mm²/s)
    D_PVS_radial_range = np.linspace(2e-4, 3e-3, 100)    # Radial diffusivity (mm²/s)
    )

genDiffusivities(
    D_PVS_axial_range   = np.linspace(5e-4, 3e-3, 100),  # Axial diffusivity (mm²/s)
    D_PVS_radial_range  = np.linspace(2e-4, 1.5e-3, 100)  # Radial diffusivity (mm²/s)
)

genAlphaVolume(
    alpha_range = np.deg2rad(np.linspace(0, 90, 100)),       # White matter angles (0° to 90°)
    f_PVS_range = np.linspace(0, 1, 100),                # Volume fraction (0 to 1) 
)

visualiseTensors()




