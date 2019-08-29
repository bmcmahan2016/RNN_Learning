# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import numpy as np
#import utils
import torch
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import matplotlib
import time

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, var=0.045):

        super(RNN, self).__init__()  # necessary for backprop to work
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #create an activity tensor that will hold a history of all hidden states
        self.activity_tensor = np.zeros((50))
        self.targets=[]
        self.losses = 0     #trainer should update this with a list of training losses
        self.model_name = 'UNSPECIFIED MODEL'   #trainer should update this
        self.pca = []

        #weight matrices for the RNN are initialized with random normal
        self.J = {
        'in' : (torch.rand(hidden_size, input_size)-0.5)*2,
        'rec' : var*torch.randn(hidden_size, hidden_size),
        'out' : (torch.rand(output_size, hidden_size)-0.5)*2
        }


    def AssignWeights(self, Jin, Jrec, Jout):
        '''
        AssignWeights will manually asign the network weights to values specified by
        Jin, Jrec, and Jout. Inputs may either be Numpy arrays or Torch tensors. The 
        model weights are maintained as Torch tensors.
        '''
        if torch.is_tensor(Jin):
            self.J['in'] = Jin
        else:
            self.J['in'] = torch.from_numpy(Jin).float()
        if torch.is_tensor(Jrec):
            self.J['rec'] = Jrec
        else:
            self.J['rec'] = torch.from_numpy(Jrec).float()
        if torch.is_tensor(Jout):
            self.J['out'] = Jout
        else:
            self.J['out'] = torch.from_numpy(Jout).float()
    def UpdateHidden(self, inpt, hidden, dt):
        hidden_next = dt*torch.matmul(self.J['in'], inpt) + \
        dt*torch.matmul(self.J['rec'], (1+torch.tanh(hidden))) + \
        (1-dt)*hidden

        return hidden_next

    def forward(self, inpt, hidden, dt):
        '''computes the forward pass activations'''
        #ensure the input is a torch tensor
        if torch.is_tensor(inpt) == False:
            inpt = torch.from_numpy(inpt).float()
        #inpt must have shape (1,1)
        #print('input:', inpt)
        inpt = inpt.reshape(self.input_size,1)
        #print(inpt)
        #time.sleep(1)
        #print('success!')
        #assert False
        #compute the forward pass
        hidden_next = self.UpdateHidden(inpt, hidden, dt)
        #dt*torch.matmul(self.J['in'], inpt) + \
        #dt*torch.matmul(self.J['rec'], (1+torch.tanh(hidden))) + \
        #(1-dt)*hidden
        output = torch.matmul(self.J['out'], torch.tanh(hidden_next))
        #return the RNN hidden state and the output
        return output, hidden_next

    def feed(self, inpt_data, dt=0.1, return_hidden=False, add_recc_noise=False):
        '''
        feed is a method that can be used for feeding input data 
        into an RNN without needing to call a trainer object (for 
        instance at test time)
        '''
        output_trace = []
        hidden_trace = []
        hidden = self.init_hidden()     #initializes a hidden state prior to feeding data
        for t_step in range(len(inpt_data)):
            output, hidden = self.forward(inpt_data[t_step], hidden, dt)
            if add_recc_noise:
                hidden += 0.1*torch.randn(hidden.shape)
            if return_hidden:
                hidden_trace.append(hidden.detach().numpy().reshape(-1,1))
            output_trace.append(torch.tanh(output))
        if return_hidden:
            return output_trace, hidden_trace
        return output_trace

    def save(self, model_name):
        '''saves current RNN'''
        model_name = model_name+'.pt'
        torch.save({'weights': self.J, 'activities': self.activity_tensor, 'targets': self.targets, 'pca': self.pca, 'losses': self.losses}, model_name)

    def load(self, model_name):
        '''loads weights of previously saved model'''
        model_name = model_name+'.pt'
        model_dict = torch.load(model_name)
        self.J = model_dict['weights']
        self.activity_tensor = model_dict['activities']
        self.targets = model_dict['targets']
        self.pca = model_dict['pca']
        self.losses = model_dict['losses']
        print('\n\nRNN model succesfully loaded ...')


    # maybe I should consider learning the initial state?
    def init_hidden(self):
        return 2*(torch.rand(self.hidden_size, 1)-0.5) 

    def test(self, input):
        pass
        '''
        hidden = torch.normal(mean=0, std=0.0375 * torch.ones(self.hidden_size, ))  # fixing here so that the randomness doesn't come from the initialzation of the hidden state
        output = np.zeros((852, 2))  # change this
        for i in range(input.shape[0]):
            output_temp, hidden = self.forward(input[i, :], hidden, self.Jin, self.Jrec, self.Jout)  # in the other case it was input[i, :]... just be careful
            output[i] = output_temp
        return output
        '''

    def VisualizeWeightClusters(self, neuron_sorting, p):
        cmap=matplotlib.cm.bwr
        weight_matrix=self.J['rec'].detach().numpy()
        weights_ordered = weight_matrix[:,neuron_sorting]
        weights_ordered = weights_ordered[neuron_sorting, :]
        #average four clusters
        C11 = np.mean( weights_ordered[:p, :p] )
        C12 = np.mean( weights_ordered[:p, p:] )
        C21 = np.mean( weights_ordered[p:, :p] )
        C22 = np.mean( weights_ordered[p:, p:] )
        weight_clusters = np.array([[C11, C12],[C21, C22]])
        plt.imshow(weight_clusters, cmap=cmap, vmin=-0.1, vmax=0.1)
        plt.title('Clustered Weight Matrix')

    def VisualizeWeightMatrix(self, neuron_sorting=[]):
        #pass
        #plt.figure()
        cmap=matplotlib.cm.bwr
        if len(neuron_sorting) != 0:
            weight_matrix=self.J['rec'].detach().numpy()
            weights_ordered = weight_matrix[:,neuron_sorting]
            weights_ordered = weights_ordered[neuron_sorting, :]
            plt.imshow(weights_ordered, cmap=cmap, vmin=-0.1, vmax=0.1)
            #plt.imshow(weights_ordered, cmap=cmap, vmin=np.min(weight_matrix), vmax=np.max(weight_matrix))
            plt.title('Ordered by Neuron Factor')
        else:
            weight_matrix = self.J['rec'].detach().numpy()
            plt.imshow(weight_matrix, cmap=cmap, vmin=-0.1, vmax=0.1)
            plt.title('Network Weight Matrix')
        plt.ylabel('Postsynaptic Neuron')
        plt.xlabel('Presynaptic Neuron')

    def GetF(self):
        W_rec = self.J['rec'].data.detach()
        W_in = self.J['in'].data.detach()

        def master_function(inpt):
            dt = 0.1
            sizeOfInput = len(inpt)
            print('sizeOfInput', sizeOfInput)
            inpt = torch.tensor(inpt).float().reshape(sizeOfInput,1)
            return lambda x: np.squeeze( ( self.UpdateHidden(inpt, torch.from_numpy(x.reshape(self.hidden_size,1)).float(), dt) -torch.from_numpy(x.reshape(self.hidden_size,1)).float()).detach().numpy() )
            #return lambda x: np.matmul(W_rec, 1+np.tanh(x)) + np.matmul(W_in, np.array([inpt])) - x

        return master_function
        
    def plotLosses(self):
        plt.plot(self.losses)
        plt.ylabel('Loss')
        plt.xlabel('Trial')
        plt.title(self.model_name)

    # def analyze():

# rnn trainGenetic
# define outputs

###########################################################
#DEBUG RNN CLASSS
###########################################################
if __name__ == '__main__':

    #hyper-parameters for RNN
    input_size=1
    hidden_size=50
    output_size=1

    #create an RNN instance
    rnn_inst = RNN(input_size, hidden_size, output_size, var=0.01)

    #verify model parameters
    print('\n\nMODEL PARAMETERS ...\n')
    for param in rnn_inst.parameters():
        print('parameter:', param.shape, '\n')
    
    #verify network forward pass
    inpt = np.random.randn(1)
    print('\n\nCOMPUTING FORWARD PASS...\n')
    hidden = rnn_inst.init_hidden()
    output, hidden = rnn_inst.forward(inpt, hidden, 0.1)
    print('network output:', output)
    print('\nupdated network hidden state:\n', hidden)

    print('\nTesting Master Function Generator...\n')
    F = rnn_inst.GetF()
    my_func = F(1)
    x=np.random.rand(50,1)
    print('output:', my_func(x).shape)




