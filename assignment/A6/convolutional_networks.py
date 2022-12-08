"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
You are NOT allowed to use torch.nn ops, unless otherwise specified.
"""
import torch

from common.helpers import softmax_loss
from common import Solver


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modify the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ######################################################################
        # TODO: Implement the convolutional forward pass.                    #
        # Hint: You can use function torch.nn.functional.pad for padding.    #
        # You are NOT allowed to use anything in torch.nn in other places.   #
        ######################################################################
        # Replace "pass" statement with your code
        pad = conv_param['pad']
        stride = conv_param['stride']
        x_padded = torch.nn.functional.pad(x, (pad,pad,pad,pad), 'constant', 0).to(x.device) #pad 2dim
        N,C,H,W = x.shape
        F,C,HH,WW = w.shape
        Ho = 1 + (H + 2 * pad - HH) // stride
        Wo = 1 + (W + 2 * pad - WW) // stride
        out = torch.zeros((N,F,Ho,Wo), dtype=x.dtype).to(x.device)
        kernel = w.flatten(start_dim=1) # (F, C*HH*WW)
        for i in range(Ho):
            for j in range(Wo):
                xi, xj = i*stride, j*stride
                patch = x_padded[:, :, xi:xi+HH, xj:xj+WW].flatten(start_dim=1) # (N, C*HH*WW)
                out[:,:,i,j] = patch.mm(kernel.T) + b # (N, F)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ######################################################################
        # TODO: Implement the convolutional backward pass.                   #
        # Hint: You can use function torch.nn.functional.pad for padding.    #
        # You are NOT allowed to use anything in torch.nn in other places.   #
        ######################################################################
        # Replace "pass" statement with your code
        x, w, b, conv_param = cache
        pad = conv_param['pad']
        stride = conv_param['stride']
        x_padded = torch.nn.functional.pad(x, (pad,pad,pad,pad), 'constant', 0).to(x.device) #pad 2dim (N,C,...)
        
        #dx
        N,F,Ho,Wo = dout.shape
        F,C,HH,WW = w.shape
        N,C,H,W = x.shape
        dx_padded = torch.zeros_like(x_padded, dtype=x.dtype).to(x.device)
        dw = torch.zeros_like(w).to(w.device)
        for i in range(Ho):
            for j in range(Wo):
                xi, xj = i*stride, j*stride
                dout_one = dout[:, :, i, j] #(N,F)

                #dx
                dot_v = torch.tensordot(dout_one, w, dims=([1], [0])) #(N,C,HH,WW) = (N,F) * (F,C,HH,WW)
                dx_padded[:, :, xi:xi+HH, xj:xj+WW] += dot_v #(N,C,HH,WW)

                #dw
                patch = x_padded[:, :, xi:xi+HH, xj:xj+WW] #(N,C,HH,WW)
                dw += torch.tensordot(dout_one, patch, dims=([0], [0])) #(F,C,HH,WW) = (N,F) * (N,C,HH,WW)

        dx = dx_padded[:,:,pad:-pad,pad:-pad]
        
        #db
        db = dout.sum((0,2,3))
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ######################################################################
        # TODO: Implement the max-pooling forward pass.                      #
        ######################################################################
        # Replace "pass" statement with your code
        PH, PW = pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        N,C,H,W = x.shape

        Ho = 1 + (H - PH) // stride
        Wo = 1 + (W - PW) // stride
        out = torch.zeros((N,C,Ho, Wo), dtype=x.dtype).to(x.device)
        for i in range(Ho):
            for j in range(Wo):
                xi, xj = i * stride, j * stride
                patch = x[:,:,xi:xi+PH,xj:xj+PW]
                patch_max = patch.amax(axis=(2,3))
                out[:,:,i,j] = patch_max

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        ######################################################################
        # TODO: Implement the max-pooling backward pass.                     #
        ######################################################################
        # Replace "pass" statement with your code
        N,C,Ho,Wo = dout.shape
        x, pool_param = cache
        PH, PW = pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        N,C,H,W = x.shape
        dx = torch.zeros((N,C,H,W), dtype=x.dtype).to(x.device)
        for i in range(Ho):
            for j in range(Wo):
                xi, xj = i * stride, j * stride
                patch = x[:,:,xi:xi+PH,xj:xj+PW] #(N,C,PH,PW)
                patch_max = patch.amax(axis=(2,3)).view(N,C,1,1) #(N,C,1,1). for broadcasting
                patch_mask = (patch == patch_max)
                dx[:,:,xi:xi+PH,xj:xj+PW] = patch_mask * dout[:,:,i,j].view(N,C,1,1)
                
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights and biases for three-layer convolutional  #
        # network. Weights should be initialized from the Gaussian           #
        # distribution with the mean of 0.0 and the standard deviation of    #
        # weight_scale; biases should be initialized to zero. All weights    #
        # and biases should be stored in the dictionary self.params.         #
        # Store weights and biases for the convolutional layer using the     #
        # keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the weights and     #
        # biases of the hidden linear layer, and keys 'W3' and 'b3' for the  #
        # weights and biases of the output linear layer.                     #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        #conv - relu - 2x2 max pool - linear - relu - linear - softmax
        #data that have shape (N, C, H, W)
        C, H, W = input_dims
        HH, WW = filter_size, filter_size
        F = num_filters
        # conv - relu - 2x2 max pool
        self.params['W1'] = torch.randn((F,C,HH,WW), dtype=dtype).to(device) * weight_scale
        self.params['b1'] = torch.zeros(F, dtype=dtype).to(device)

        # linear - relu
        self.params['W2'] = torch.randn((F*(H//2)*(W//2), hidden_dim), dtype=dtype).to(device) * weight_scale
        self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype).to(device)
        
        # linear - softmax
        self.params['W3'] = torch.randn((hidden_dim, num_classes), dtype=dtype).to(device) * weight_scale
        self.params['b3'] = torch.zeros(num_classes, dtype=dtype).to(device)
        
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Pass conv_param to the forward pass for the convolutional layer.
        # Padding and stride chosen to preserve the input spatial size.
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # Pass pool_param to the forward pass for the max-pooling layer.
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        # Use sandwich layers if Linear or Conv layers followed by ReLU      #
        # and/or Pool layers for efficient implementation.                   #
        ######################################################################
        # Replace "pass" statement with your code
        #conv - relu - 2x2 max pool - linear - relu - linear - softmax
        h, cache1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        h, cache2 = Linear_ReLU.forward(h, W2, b2)
        scores, cache3 = Linear.forward(h, W3, b3)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ######################################################################
        # TODO: Implement the backward pass for three-layer convolutional    #
        # net, storing the loss and gradients in the loss and grads.         #
        # Compute the data loss using softmax, and make sure that grads[k]   #
        # holds the gradients for self.params[k]. Don't forget to add        #
        # L2 regularization!                                                 #
        # NOTE: To ensure your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization includes    #
        # a factor of 0.5 to simplify the expression for the gradient.       #
        ######################################################################
        # Replace "pass" statement with your code
        loss, dout = softmax_loss(scores, y)
        dout, grads['W3'], grads['b3'] = Linear.backward(dout, cache3)
        dout, grads['W2'], grads['b2'] = Linear_ReLU.backward(dout, cache2)
        dout, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dout, cache1)

        # L2
        for i in range(3, 0, -1):
            lbl = str(i)
            loss += self.reg * 0.5 * torch.sum(self.params['W'+lbl]**2) #L2
            grads['W'+lbl] += self.reg * self.params['W'+lbl] #L2
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this dtype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        ######################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights,  #
        # biases, and batchnorm scale and shift parameters should be stored  #
        # in the dictionary self.params. Weights for Conv and Linear layers  #
        # should be initialized from the Gaussian distribution with the mean #
        # of 0.0 and the standard deviation of weight_scale; biases should   #
        # be initialized to zero.                                            #
        # Batchnorm scale (gamma) and shift (beta) parameters should be      #
        # initialized to ones and zeros, respectively.                       #
        ######################################################################
        # Replace "pass" statement with your code
        # All convolution layers will use kernel size 3 and padding 1 to preserve the feature map size, 
        # {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear
        C,H,W = input_dims
        KS = 3
        for i, F in enumerate(num_filters):
            lbl = str(i+1)
            if weight_scale == 'kaiming':
                self.params['W'+lbl] = kaiming_initializer(Din=C, Dout=F, K=KS, relu=True, device=device, dtype=dtype)
            else:
                self.params['W'+lbl] = torch.randn((F,C,KS,KS), dtype=dtype).to(device) * weight_scale
            self.params['b'+lbl] = torch.zeros(F, dtype=dtype).to(device)
            if self.batchnorm:
                self.params['G'+lbl] = torch.ones(F, dtype=dtype).to(device)
                self.params['B'+lbl] = torch.ones(F, dtype=dtype).to(device)
            C = F
        hidden_dim = F * (H * W) // (4**len(max_pools))
        if weight_scale == 'kaiming':
            self.params['W'+str(self.num_layers)] = kaiming_initializer(Din=hidden_dim, Dout=num_classes, relu=False, device=device, dtype=dtype)
        else:
            self.params['W'+str(self.num_layers)] = torch.randn((hidden_dim, num_classes), dtype=dtype, device=device) * weight_scale
        self.params['b'+str(self.num_layers)] = torch.zeros(num_classes, dtype=dtype, device=device)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for DeepConvNet, computing the    #
        # class scores for X and storing them in the scores variable.        #
        # Use sandwich layers if Linear or Conv layers followed by ReLU      #
        # and/or Pool layers for efficient implementation.                   #
        ######################################################################
        # Replace "pass" statement with your code
        # {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear - softmax
        LayerClass = [
            [Conv_ReLU,             Conv_ReLU_Pool          ],
            [Conv_BatchNorm_ReLU,   Conv_BatchNorm_ReLU_Pool]
        ]
        caches = []
        h = X
        for i in range(self.num_layers-1):
            lbl = str(i+1)
            args = {
                'x': h,
                'w': self.params['W'+lbl],
                'b': self.params['b'+lbl],
                'conv_param': conv_param
            }
            if self.batchnorm:
                args['gamma'] = self.params['G'+lbl]
                args['beta'] = self.params['B'+lbl]
                args['bn_param'] = self.bn_params[i]
            if i in self.max_pools:
                args['pool_param'] = pool_param
            cls = LayerClass[int(self.batchnorm)][int(i in self.max_pools)]
            h, cache = cls.forward(**args)
            caches.append(cache)
        lbl = str(self.num_layers)
        scores, cache = Linear.forward(h, self.params['W'+lbl], self.params['b'+lbl])
        caches.append(cache)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ######################################################################
        # TODO: Implement the backward pass for the DeepConvNet, storing the #
        # loss and gradients in the loss and grads variables.                #
        # Compute the data loss using softmax, and make sure that grads[k]   #
        # holds the gradients for self.params[k]. Don't forget to add        #
        # L2 regularization!                                                 #
        # NOTE: To ensure your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization includes    #
        # a factor of 0.5 to simplify the expression for the gradient.       #
        ######################################################################
        # Replace "pass" statement with your code
        loss, dout = softmax_loss(scores, y)
        lbl = str(self.num_layers)
        dout, grads['W'+lbl], grads['b'+lbl] = Linear.backward(dout, caches.pop())
        for i in range(self.num_layers-2, -1, -1):
            lbl = str(i+1)
            cls = LayerClass[int(self.batchnorm)][int(i in self.max_pools)]
            ret = cls.backward(dout, caches.pop())
            dout, grads['W'+lbl], grads['b'+lbl] = ret[0], ret[1], ret[2]
            if self.batchnorm:
                grads['G'+lbl], grads['B'+lbl] = ret[3], ret[4]

        # L2
        for i in range(self.num_layers):
            lbl = str(i+1)
            loss += self.reg * 0.5 * torch.sum(self.params['W'+lbl]**2) #L2
            grads['W'+lbl] += self.reg * self.params['W'+lbl] #L2

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ##########################################################################
    # TODO: Change weight_scale and learning_rate so your model achieves     #
    # 100% training accuracy within 30 epochs.                               #
    ##########################################################################
    # Replace "pass" statement with your code
    
    ##########################################################################
    weight_scale = 5e-1
    learning_rate = 1e-3
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return weight_scale, learning_rate


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initialization); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ######################################################################
        # TODO: Implement the Kaiming initialization for linear layer.       #
        # The weight_scale is sqrt(gain / fan_in), where gain is 2 if ReLU   #
        # is followed by the layer, or 1 if not, and fan_in = Din.           #
        # The output should be a tensor in the designated size, dtype,       #
        # and device.                                                        #
        ######################################################################
        # Replace "pass" statement with your code
        fan_in = Din
        weight_scale = torch.tensor(gain / fan_in).sqrt()
        weight = torch.randn((Din, Dout), dtype=dtype).to(device) * weight_scale
        # torch.randn((hidden_dim, num_classes), dtype=dtype, device=device) * weight_scale
        
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    else:
        ######################################################################
        # TODO: Implement Kaiming initialization for convolutional layer.    #
        # The weight_scale is sqrt(gain / fan_in), where gain is 2 if ReLU   #
        # is followed by the layer, or 1 if not, and fan_in = Din * K * K.   #
        # The output should be a tensor in the designated size, dtype,       #
        # and device.                                                        #
        ######################################################################
        # Replace "pass" statement with your code
        fan_in = Din * K * K
        weight_scale = torch.tensor(gain / fan_in).sqrt()
        weight = torch.randn((Dout, Din, K, K), dtype=dtype).to(device) * weight_scale
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    return weight


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    ##########################################################################
    # TODO: Train the best DeepConvNet on CIFAR-10 within 60 seconds.        #
    ##########################################################################
    # Replace "pass" statement with your code
    input_dims = data_dict['X_train'].shape[1:]

    model = DeepConvNet(input_dims=input_dims, num_classes=10,
                        num_filters=[32,64,192],
                        max_pools=[1,2],
                        weight_scale='kaiming',
                        reg=1e-7,
                        dtype=dtype, device=device)

    solver = Solver(model, data_dict,
                    num_epochs=4, batch_size=64,
                    update_rule=Solver.adam,
                    lr_decay=5e-1,
                    optim_config={
                        'learning_rate': 2e-3,
                    },
                    print_every=300, device=device)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return solver


##################################################################
#            Fast Implementations and Sandwich Layers            #
##################################################################


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        layer = torch.nn.Linear(*w.shape)
        layer.weight = torch.nn.Parameter(w.T)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx.flatten(start_dim=1))
        cache = (x, w, b, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, w, b, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach().T
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dw = torch.zeros_like(layer.weight).T
            db = torch.zeros_like(layer.bias)
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        layer = torch.nn.ReLU()
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs a linear transform followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output of the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dw = torch.zeros_like(layer.weight)
            db = torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool convenience layer.
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class BatchNorm(object):
    func = torch.nn.BatchNorm1d

    @classmethod
    def forward(cls, x, gamma, beta, bn_param):
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)
        D = x.shape[1]
        running_mean = \
            bn_param.get('running_mean',
                         torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = \
            bn_param.get('running_var',
                         torch.ones(D, dtype=x.dtype, device=x.device))

        layer = cls.func(D, eps=eps, momentum=momentum,
                         device=x.device, dtype=x.dtype)
        layer.weight = torch.nn.Parameter(gamma)
        layer.bias = torch.nn.Parameter(beta)
        layer.running_mean = running_mean
        layer.running_var = running_var
        if mode == 'train':
            layer.train()
        elif mode == 'test':
            layer.eval()
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (mode, x, tx, out, layer)
        # Store the updated running means back into bn_param
        bn_param['running_mean'] = layer.running_mean.detach()
        bn_param['running_var'] = layer.running_var.detach()
        return out, cache

    @classmethod
    def backward(cls, dout, cache):
        mode, x, tx, out, layer = cache
        try:
            if mode == 'train':
                layer.train()
            elif mode == 'test':
                layer.eval()
            else:
                raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
            out.backward(dout)
            dx = tx.grad.detach()
            dgamma = layer.weight.grad.detach()
            dbeta = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dgamma = torch.zeros_like(layer.weight)
            dbeta = torch.zeros_like(layer.bias)
        return dx, dgamma, dbeta


class SpatialBatchNorm(BatchNorm):
    func = torch.nn.BatchNorm2d


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D1, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
