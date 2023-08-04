import numpy as np
import scipy as sp
from scipy.stats import zscore, kurtosis
from sklearn.decomposition import PCA
from .helpers import negentropy


class constrainedICA():
    """A constrained ICA algorithm for semi-blind source separation.
    
    Parameters
    ----------
    lr : float, default=0.1
        Learning rate for the gradient descent optimization.
        
    thres : float, default=0.5
        Similarity threshold for the constraint.
        
    mu_init : float, default=0
        Initial Lagrangian parameter.
        
    gamma : float, default=1
        Update step parameter for the constraint.
        
    stop_crit : float, default=1e-8
        Stop criterion.
        
    max_iter : int, default=10000
        Maximum number of iteration.
        
    whiten : bool, default=True
        Whiten the input signals.
        
    zscore_refs : bool, default=True
        Z-score reference signals.
        
    decore_w : bool, default=True
        Decorrelate weight matrix W at each iteration.
        
    verbose : int, default=10
        Print infos every 'verbose' iteration.
        
    constraint : bool, default=True
        Run constrained or unconstrained ICA.
        
    n_pca_comp : int, default=None
        Number of PCA components the signals are projected on
        prior to running ICA.
        
    n_ica_comp : int, default=None
        Number of ICA components/source signals that are
        estimated.
        
    annealing_lr : int, default=None
        Use annealing learning rate for learning.
        - if annealing_lr is not None, deacrease every 
        'annealing_lr'-th iteration the learning rate by 
        a factor of 10.
        
    Examples
    -------
    from cICA.functions import constrainedICA
    cICA = constrainedICA()
    Y, W, history = cICA.fit_transform(X, Refs)
    """
    
    def __init__(
        self,
        lr=0.1, 
        thres=0.5, 
        mu_init=0, 
        gamma=1, 
        stop_crit=1e-8, 
        max_iter=10000, 
        whiten=True,
        zscore_Refs=True,
        decor_w=True, 
        verbose=10, 
        constraint=True,
        n_pca_comp=None,
        n_ica_comp=None,
        annealing_lr=None,
        obj_func='negentr'
    ):
        super().__init__()
        self.lr = lr
        self.thres = thres
        self.mu_init = mu_init
        self.gamma = gamma
        self.stop_crit = stop_crit
        self.max_iter = max_iter
        self.whiten = whiten
        self.zscore_Refs = zscore_Refs
        self.decor_w = decor_w
        self.verbose = verbose
        self.constraint = constraint
        self.n_pca_comp = n_pca_comp
        self.n_ica_comp = n_ica_comp
        self.obj_func = obj_func
        self.annealing_lr = annealing_lr
        
    def fit_transform(self, X, Refs):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples)
            Mixtures of signals, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Refs : array-like of shape (n_refs, n_samples)
            Reference signals, where `n_samples` is the number of samples
            and `n_refs` is the number of features.

        Returns
        -------
        Y : ndarray of shape (n_ica_comp, n_samples)
            Sources matrix.
            
        W : ndarray of shape (n_ica_comp, n_features)
            Demixing matrix.
        """
        if self.n_pca_comp is not None or self.whiten is True:
            print('Apply PCA with n_pca_comp: {}, whiten: {}.\n'.format(self.n_pca_comp, self.whiten))
            self.pca = PCA(n_components=self.n_pca_comp, whiten=self.whiten)
            X = self.pca.fit_transform(X.T).T  # Sklearn wants inputs with shape (n_samples, n_features).
            
        if self.zscore_Refs is True:
            Refs = zscore(Refs, axis=1)

        self.n_feat, self.n_samp = X.shape
        self.n_refs, _ = Refs.shape
            
        if self.n_ica_comp is None:
            self.n_ica_comp = self.n_feat
            
        # Initialize empty references, in case n_ica_comp > n_refs.
        Refs_full = np.zeros((self.n_ica_comp, self.n_samp))
        for n, ref in enumerate(Refs):
            Refs_full[n] = ref
            
        print('Apply ICA with n_ica_comp: {}, n_refs: {}.\n'.format(self.n_ica_comp, self.n_refs))
            
        self.mu = np.repeat(self.mu_init, self.n_ica_comp)
        self.gamma = np.repeat(self.gamma, self.n_ica_comp)
        self.thres = np.repeat(self.thres, self.n_ica_comp)

        # Initialize W.
        W = np.ones((self.n_ica_comp, self.n_feat)) + \
            0.01 * np.random.uniform(size=(self.n_ica_comp, self.n_feat))

        # Select objective function for ICA.
        if self.obj_func == 'negentr':
            dObj_func = self.dNegEntr
        elif self.obj_func == 'infomax':
            dObj_func = self.dInfomax
        elif self.obj_func == 'ext_infomax':
            dObj_func = self.dExt_infomax
            
        history = []
        # Start training.
        for n_iter in range(1, self.max_iter):

            # Compute estimations for source signals.
            Y = W @ X

            # Update Lagrange multiplier.
            similarity = np.mean(Y * Refs_full, axis=1)
            g = self.thres - similarity
            mu = np.maximum(0, self.mu + self.gamma * g)

            # Compute first order derivative.
            if self.constraint is True:
                dL = dObj_func(Y, X, W) + np.diag(mu) @ (Refs_full @ X.T) / self.n_feat 
            else:
                dL = dObj_func(Y, X, W)

            # Update weights.
            W_old = W
            W = W + self.lr * dL

            # Normalize weights.
            W_s = np.linalg.norm(W, axis=1, keepdims=True)
            W = W / W_s

            # Decorrelate weights if negentropy obj_func is used.
            if self.decor_w is True and self.obj_func == 'negentr':
                W = self.sym_decorrelation(W)
                
            # Check stopping criterion.
            W_diff = np.max(np.abs(np.abs(np.einsum("ij,ij->i", W_old, W)) - 1))

            # Log history.
            for n_IC in range(Y.shape[0]):
                history.append({'iteration': n_iter, 'IC': n_IC, 'W_diff': W_diff, 
                                'mu': mu[n_IC], 'similarity': similarity[n_IC], 
                                'negentropy': negentropy(Y[[n_IC], :])[0],
                                'kurtosis': kurtosis(Y[n_IC])}) 
                
            # Print infos.
            if self.verbose is not False:
                if (n_iter % self.verbose) == 0:
                    self.print_status(n_iter, W_diff, mu, similarity, Y)
                
            # Annealing learning rate.
            if self.annealing_lr is not None and (n_iter % self.annealing_lr) == 0:
                self.lr = self.lr * 0.1
                print('Lowering learning rate to: {:.4f}\n'.format(self.lr))

            # Check stop criterion.
            if W_diff < self.stop_crit:
                print('Converged at iteration: #{:05d}.'.format(n_iter))
                self.print_status(n_iter, W_diff, mu, similarity, Y)
                break
                
        Y = W @ X
        
        # Compute final unmixing matrix W for decomposing original data matrix X into ICs.
        if self.n_pca_comp is not None or self.whiten is True:
            if self.whiten:
                # 'pca.explained_variance_' is computed in sklearn as:
                # U, S, Vt = linalg.svd(X, full_matrices=False)
                # explained_variance_ = (S**2) / (n_samples - 1)
                Q = self.pca.components_ / np.sqrt(np.expand_dims(self.pca.explained_variance_, 1))
            else:
                Q = self.pca.components_
                
            W = W @ Q
            
        return Y, W, history

    def dNegEntr(self, Y, X, W=None):
        """Derivative of approximated Negentropy.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples)
            Mixtures of signals, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            
        Y : array-like of shape (n_features, n_samples)
            Estimated sources, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            
        W : ndarray of shape (n_ica_comp, n_features)
            Demixing matrix (not used in this objective function).
        """
        rou = negentropy(Y)
        return np.diag(np.sign(rou)) @ (np.tanh(Y) @ X.T) / self.n_samp

    def dInfomax(self, Y, X, W):
        """Derivative of objective Function for Infomax algorithm.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples)
            Mixtures of signals, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            
        Y : array-like of shape (n_features, n_samples)
            Estimated sources, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            
        W : ndarray of shape (n_ica_comp, n_features)
            Demixing matrix (not used in this objective function).
        """
        return (-2 * np.tanh(Y) @ X.T) / self.n_samp + np.linalg.pinv(W.T)

    def dExt_infomax(self, Y, X, W):
        """Derivative of objective Function for extended Infomax algorithm.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples)
            Mixtures of signals, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            
        Y : array-like of shape (n_features, n_samples)
            Estimated sources, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            
        W : ndarray of shape (n_ica_comp, n_features)
            Demixing matrix (not used in this objective function).
        """
        G = []
        for y in Y:
            y_i = np.mean(-np.tanh(y) * y + 1 - np.power(np.tanh(y), 2))
            if y_i > 0:  # For supergaussian densities.
                G.append(-2 * np.tanh(y))
            else:  # For subgaussian densities.
                G.append(np.tanh(y) - y)
        G = np.stack(G)
        return (G @ X.T) / self.n_samp + np.linalg.pinv(W.T)

    def sym_decorrelation(self, W):
        """Symmetric decorrelation
        i.e. W <- (W * W.T) ^{-1/2} * W
        Based on: https://github.com/scikit-learn/scikit-learn       
        
        Parameters
        ----------     
        W : ndarray of shape (n_ica_comp, n_features)
            Demixing matrix.
        """
        s, u = sp.linalg.eigh(np.dot(W, W.T))
        # Avoid sqrt of negative values because of rounding errors. Note that
        # np.sqrt(tiny) is larger than tiny and therefore this clipping also
        # prevents division by zero in the next step.
        s = np.clip(s, a_min=np.finfo(W.dtype).tiny, a_max=None)

        # u (resp. s) contains the eigenvectors (resp. square roots of
        # the eigenvalues) of W * W.T
        return np.linalg.multi_dot([u * (1.0 / np.sqrt(s)), u.T, W])
    
    def print_status(self, n_iter, W_diff, mu, similarity, Y):
        """Prints status of algorithm.     
        
        Parameters
        ----------     
        n_iter : int
            Number of iteration.
            
        W : ndarray of shape (n_ica_comp, n_features)
            Difference of demixing matrix.
            
        similarity : ndarray of shape (n_refs)
            Similarities to reference signals.  
            
        mu : ndarray of shape (n_refs)
            Lagrange parameters.  
            
        Y : array-like of shape (n_features, n_samples)
            Estimated sources, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        """
        message = 'Iteration #{:05d}: Change W: {:.8f} \n  Lag. mult.: {} \n  Similarity: {} \n  Negentropy: {} \n  Kurtosis:  {} \n'
        print(message.format(n_iter, 
                             W_diff, 
                             np.array2string(mu, precision=4), 
                             np.array2string(similarity, precision=4),
                             np.array2string(negentropy(Y), precision=4), 
                             np.array2string(kurtosis(Y, axis=1), precision=4)))