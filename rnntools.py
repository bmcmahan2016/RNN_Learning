# TODO: perhaps put in RNN class; integrate a bit better code
'''
This file contains some useful functions that can be applied to trained RNNs
Authors: Brandon McMahan and Michael Kleinman
'''


import facilities as fac
import pdb
import matplotlib.pyplot as plt
import numpy as np
import tensortools as tt
import matplotlib
from task.williams import Williams

# retrieve data
#datapath = 'data/miconi/'
# Jrec = fac.Retrieve('Jrec.p', datapath)
# Jin = fac.Retrieve('Jin.p', datapath)
# Jout = fac.Retrieve('Jout.p', datapath)
# inps = fac.Retrieve('inps_save.p', datapath)
#activity_tensor = fac.Retrieve('activity_tensor.p', datapath)


# histogram of weights
def plotHist(Jrec, num_bins):
    data = Jrec.reshape(Jrec.shape[0] * Jrec.shape[1], )
    plt.hist(data, num_bins, color='blue', alpha=0.6)
    plt.show()


# eigenspectrum of Wrec
def plotEigenvals(Jrec):
    e, _ = np.linalg.eig(Jrec)
    plt.plot(e.real, e.imag, '.')
    plt.xlabel('real eigenvalues', fontSize=14)
    plt.ylabel('imaginary eigenvalues', fontSize=14)
    plt.show()


# TCA
def plotTCs(activity_tensor, targets, R, title=''):
    U = tt.cp_als(activity_tensor, rank=R, verbose=False)
    tt.plot_factors(U.factors, targets, plots=['scatter', 'line', 'bar'])
    plt.title(title)
    #plt.show()
    # plt.savefig('tensorFactors.pdf')
    #return neuron factors
    return U.factors[2][:,0]


# PCA
def pca(data):
    means = np.mean(data, axis=1)
    data_centered = (data.T - means).T
    evecs, evals, _ = np.linalg.svd(np.cov(data_centered))
    scores = np.dot(evecs.T, data_centered)
    return evecs, evals, scores, means


def plotPC1(data):
    evecs, evals, scores, means = pca(data.T)
    plt.plot(-scores[0, :], linewidth=2, color='black')
    plt.show()


# variance explained by PCs
def plotVariance(data, components):
    evecs, evals, scores, means = pca(data.T)
    plt.plot(evals[:components] / np.sum(evals), '.', color='blue')
    plt.ylabel('Variance Explained')
    plt.xlabel('Component')
    plt.show()


# single unit
def plotSingleUnit(data, unit1, unit2, unit3):
    plt.plot(data[:, unit1], linewidth=4, c='r', label='')
    plt.plot(data[:, unit2], linewidth=4, c='b', label='')
    plt.plot(data[:, unit3], linewidth=4, c='k', label='')
    #plt.show()


# plot multi unit through time
def plotMultiUnit(data, normalize_cols=False):
    data = data.T
    #normalize the activities at each timestep to avoid global changes in activity
    if normalize_cols:
        #loop over each column of data (corresponding to a timepoint in trial)
        for _ in range(len(data[0])):
            data[:,_] = data[:,_] - np.mean(data[:,_])
    plt.imshow(data, cmap='hot', interpolation='nearest', vmin=-1, vmax=1, aspect='auto') #temporarily removed vmin=-1 and vmax=1
    plt.xlabel('time')
    plt.ylabel('units')
    #plt.show()

def plotPSTH(rnn_inst, neuron_idx=[], num_last=10):
    #if there is no order for neurons create a trivial one
    if len(neuron_idx) == 0:
        neuron_idx = np.linspace(0, len(rnn_inst.hidden_size), len(rnn_inst.hidden_size))
    
    targets = np.array(rnn_inst.targets)
    first_pos_trial = np.where(targets==1)[0][0]
    last_pos_trial = np.where(targets==1)[0][-num_last:]

    first_neg_trial = np.where(targets==-1)[0][0]
    last_neg_trial = np.where(targets==-1)[0][-num_last:]

    #print('first positive trial', first_pos_trial, 'last positive trial', last_pos_trial)
    #print('first negative trial', first_neg_trial, 'last negative trial', last_neg_trial)

    #get recurrent network activity tensor ordered by neuron factors
    activity_tensor = rnn_inst.activity_tensor[:,:,neuron_idx]

    #neuron activities during first positive and negative trial
    activity_fpt = activity_tensor[first_pos_trial,:,:]
    activity_fnt = activity_tensor[first_neg_trial,:,:]

    #activities in num_last positive trials
    activity_lpt = []
    for _ in range(num_last):
        activity_lpt.append(activity_tensor[last_pos_trial[_],:,:])

    #activities in num_last negative trials
    activity_lnt = []
    for _ in range(num_last):
        activity_lnt.append(activity_tensor[last_neg_trial[_],:,:])


    #plot the activity of all neurons before training
    plt.subplot(121)
    plotMultiUnit(activity_fpt)
    plt.title('First Positive Trial (Trial #{})'.format(first_pos_trial))
    plt.subplot(122)
    plotMultiUnit(activity_fnt)
    plt.title('First Negative Trial (Trial #{})'.format(first_neg_trial))
    #plt.colorbar()

    #plot num_last() trials positive and negative
    plt.figure()
    for _ in range(num_last):
        plt.figure()
        plt.subplot(121)
        plotMultiUnit(activity_lpt[_])
        plt.title('Positive Trial (Trial #{})'.format(last_pos_trial[_]))

        plt.subplot(122)
        plotMultiUnit(activity_lnt[_])
        plt.title('Negative Trial (Trial #{})'.format(last_neg_trial[_]))

        #plt.colorbar()

def plotWeights(rnn_inst, neuron_idx):
    plt.figure()
    weights = rnn_inst.J['rec'].data.numpy()
    print(weights.shape)
    weight_max = np.max(weights)
    weight_min = np.min(weights)
    cmap=matplotlib.cm.bwr
    plt.subplot(121)
    plt.imshow(weights, cmap=cmap, vmin=weight_min, vmax=weight_max)
    plt.title('Reccurent Weights')
    plt.ylabel('Post-Synaptic')
    plt.xlabel('Pre-Synaptic')

    plt.subplot(122)
    weights_ordered = weights[:, neuron_idx]
    weights_ordered = weights_ordered[neuron_idx, :]
    plt.imshow(weights_ordered, cmap=cmap, vmin=weight_min, vmax=weight_max)
    plt.title('Reccurent Weights (ordered)')
    plt.ylabel('Post-Synaptic')
    plt.xlabel('Pre-Synaptic')
    #plt.colorbar()

def plotSpeed(data):
    diff = np.diff(data, axis=0)
    result = np.linalg.norm(diff, axis=1)
    plt.plot(result, color='black', linewidth=2)
    plt.xlabel('time')
    plt.ylabel('average speed')
    plt.show()


def plotEnergy(data):
    diff = np.diff(data, axis=0)
    result = np.linalg.norm(0.5 * diff ** 2, axis=1)
    plt.plot(result, color='black', linewidth=2)
    plt.xlabel('time')
    plt.ylabel('average energy')
    plt.show()

def TestIdenticalInputs(model, conditions, cs=['r', 'b', 'k', 'g', 'y'], title=''):
    '''tests network on identical inputs'''
    plt.figure()
    for _ in range(len(conditions)):
        #create identical input data
        data = np.linspace(conditions[_], conditions[_], 400).reshape(400,1)
        output = model.feed(data)
        plt.plot(output, c=cs[_], alpha=0.8)
        #print('condition:', conditions[_], 'color', cs[_])
        additional_title = '\ninput:' + str(conditions[_]) + ' is color:' + cs[_] 
        title += additional_title
    plt.title(title)

def TestTaskInputs(model, task):
    plt.figure()
    for _ in range(10):
        inp, condition = task.GetInput()
        output = model.feed(inp)
        if condition == 1:
            plt.plot(output, c='r')
        else:
            plt.plot(output, c='b')

def TestInputs(model, variances=[2,4,8], cs=['r', 'b', 'k', 'g', 'y'], title=''):
    plt.figure()
    for _ in range(len(variances)):
        #create identical input data
        task = Williams(variance=variances[_])
        condition = 0
        while condition != 1:
            data, condition = task.GetInput()
        output = model.feed(data)
        #print('target is', condition.item())
        plt.plot(output, c=cs[_], alpha=0.8)
        #print('variance:', variances[_], 'color', cs[_])
        additional_title = '\nvariance:' + str(variances[_]) + 'is color:' + cs[_] 
        title += additional_title
    plt.title(title)

def record(model, conditions, cs=['r', 'b', 'k', 'g', 'y'], title='', print_out=False, plot_recurrent=True, add_noise=False):
    '''
    Records from recurrent neurons during an identical input sequence of length 50 while
    plotting the hidden state and activation of output neuron. Will also return hidden states 
    as a list of numpy arrays.

    ------------------
    conditions: a list of input conditions to feed the network. The number of trials is
            equal to the number of elements in conditions. 
    
    print_out: if True this will generate a figure containing a plot of the networks 
            output at each timestep
    
    plot_recurrent: if True this will plot a PSTH of the hidden state for each trial
    
    cs: colors to use for each trial
    
    title: title that will be used for network output plots, recomend setting to the 
            name of the model
    '''
    trial_data = [] #each element will be hidden trajectories for a given trial

    #create new figure if network output requested
    #if print_out:
        #create a new figure to hold outputs for each trial
        #plt.figure()
        #plot_annotated = False

    #loop over conditions to create trials
    for _ in range(len(conditions)):
        length_of_data=400   #100
        #create identical valued input data and feed it to network
        data = np.linspace(conditions[_], conditions[_], length_of_data).reshape(length_of_data,1)
        output, hidden = model.feed(data, return_hidden=True, add_recc_noise=add_noise)

        #convert output to numpy array
        for lol in range(len(output)):
            output[lol] = output[lol].detach().item()
        output = np.array(output)

        #if a plot of the network output is requested
        if print_out:
            plt.figure(101)
            plt.plot(output, c=cs[_], alpha=0.5)
            #add labels to plot if not already done
            #if not plot_annotated:
            plt.title('Activity of Output Neuron for ' +str(title)+ ' Model')
            plt.xlabel('Time in Trial')
            plt.ylabel('Activation of Output Neuron')
            plot_annotated=True

        #plot multi-unit activity
        data = np.squeeze(np.array(hidden))
        trial_data.append(data)
        if plot_recurrent:
            plt.figure()
            plotMultiUnit(data, normalize_cols=True)

    return np.array(trial_data)



# TODOs:
# PCs_2D
# PCs_3D
# optimization to find fixed points (Jonathan has code)
# input vs recurrent subspace (Goudar et al., eLife 2018)
# activity in nullspace of Wout, or 'potent space' of Wout
# perturbation experiments (deleting a node, modifying/deleting weights, etc)
# angles between eigenvectors


# data = activity_tensor[-1, :, :]
# plotSpeed(data)
# plotEnergy(data)
# plotMultiUnit(data)
# plotVariance(data, components=20)
# plotEigenvals(Jrec)
# plotHist(Jrec, num_bins=50)
#plotTCs(activity_tensor, 1)
# plotPC1(data)
# plotSingleUnit(data, unit1=21, unit2=31, unit3=41)
