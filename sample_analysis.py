from train_models import *
from perturbation_experiments import *
from task.williams import Williams
'''
NOTES
#model8 is model used in paper analysis
#model17 is for force only and is just another training instance that can also be used in paper
#model19 is without training in and out weights
#model22 were retrained with targets saved
#model76 are trained on the context task

#model99 is for genetic only and was trained and saved all activity tensors


###################################################################################

The 100 series models have been updated to save the loss tensor during training
model100 --> 2,000 training iterations (20 for genetic)
model150 --> 1,000 training iterations (15 for genetic)
model120 --> 500 training iterations (10 for genetic)
model110 --> 100 training iterations (5 for genetic)
model111 --> 1 training iteration (1 for genetic)

'''
#TrainGenetic(191, num_generations=20)
#TrainGenetic(99, num_generations=20)
#TrainFORCE(17)
#TrainHebian(22)
#TrainFORCE(22)

#1,0000 iterations
#TrainBPTT(100, num_epochs=1000)
#niave_network('bptt_model100')

#500 iterations
#TrainBPTT(150, num_epochs=500)
#niave_network('bptt_model150')

#200 iterations
#TrainBPTT(120, num_epochs=200)
#niave_network('bptt_model120')

#100 iterations
#TrainBPTT(110,num_epochs=100)
#niave_network('bptt_model110')

#1 iteration
#TrainBPTT(111, num_epochs=1)
#niave_network('bptt_model111')

#USE THIS CODE TO PLOT ANALYSIS FOR NETWORK
#niave_network('bptt_context_model76')
niave_network('force_model100')
#remove_inhibition('bptt_model8')
#remove_excitation('bptt_model8')
#niave_network('genetic_model191')
#niave_network('force_model22')
#niave_network('force_model8')
#remove_inhibition('force_model8')
#remove_excitation('force_model8')

#TestModel('force_model100')
#ContextFixedPoints('bptt_context_model76')
#plt.show()

#remove_lo_mag('bptt_model8')

'''
REMOVE all recurrent connections on hebian
'''
#niave_network('genetic_model19')
#remove_lo_mag('genetic_model8') 

plt.show()


'''
TRAIN SMALL BPTT NETWORK
rnn_inst = TrainBPTT(10, hidden_size=2, num_epochs=20_000, learning_rate=1e-5)
#perform TCA
activity_tensor = rnn_inst.activity_tensor
neuron_factor = r.plotTCs(activity_tensor, 1)
neuron_idx = np.argsort(neuron_factor)
plt.figure()
rnn_inst.VisualizeWeightMatrix(neuron_sorting=neuron_idx)
plt.figure()
rnn_inst.plotLosses()
plt.show()
'''
#TrainBPTT_context(76)