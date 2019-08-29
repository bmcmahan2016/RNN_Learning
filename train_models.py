'''
This module can be used to train models according to the following training rules
	BPTT:
	Genetic:
	Hebian:
	FORCE:

Author: Brandon McMahan
June 24, 2019
'''

import numpy as np 
from rnn import RNN 
import rnntools as r
from task.williams import Williams 
import utils
import matplotlib.pyplot as plt 
from train.bptt import Bptt 
from train.genetic import Genetic 
from train.hebian_clone import Hebian
from train.force_BM import Force
from FP_Analysis import FindFixedPoints
from task.contexttask import ContextTask

#RNN architecture
input_size = 1
hidden_size = 50
output_size = 1

#create the task
task = Williams()
context_task = ContextTask()


def TrainBPTT(identifier, hidden_size=hidden_size, num_epochs=2_000, learning_rate=1e-4):
	bptt_model = RNN(input_size, hidden_size, output_size, var=0.01)
	#train BPTT
	
	#num_epochs=2_000
	trial_length=40
	trainer=Bptt(bptt_model, task, learning_rate, num_epochs, trial_length)
	trainer.init_network()
	trainer.trainBPTT()
	F = bptt_model.GetF()
	roots, pca = FindFixedPoints(F, [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,\
				-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1], num_hidden=hidden_size)
	bptt_model.pca = pca
	bptt_model.save('bptt_model'+str(identifier))
	return bptt_model

def TrainBPTT_context(identifier, hidden_size=hidden_size, num_epochs=10_000, learning_rate=5e-5):
	bptt_model = RNN(4, hidden_size, 1, var=0.01)
	#train BPTT
	
	trial_length=40
	trainer=Bptt(bptt_model, context_task, learning_rate, num_epochs, trial_length)
	trainer.init_network()
	trainer.trainBPTT()
	#F = bptt_model.GetF()
	#roots, pca = FindFixedPoints(F, [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,\
	#			-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1], num_hidden=hidden_size)
	#bptt_model.pca = pca
	bptt_model.save('bptt_context_model'+str(identifier))
	plt.plot(bptt_model.losses)
	plt.show()
	return bptt_model

def TrainGenetic(identifier, num_generations=15):
	genetic_model = RNN(input_size, hidden_size, output_size)
	#train model using genetic algorithm
	num_pop=50
	sigma=0.01
	#num_generations=15
	trainer = Genetic(genetic_model, task, num_generations)
	trainer.trainGenetic(num_pop, sigma, batch_size=50, num_parents=5, mutation=0.1)
	F = genetic_model.GetF()
	roots, pca = FindFixedPoints(F, [[1],[0.9],[0.8],[0.7],[0.6],[0.5],[0.4],[0.3],[0.2],[0.1],\
				[-0.1],[-0.2],[-0.3],[-0.4],[-0.5],[-0.6],[-0.7],[-0.8],[-0.9],[-1]])
	genetic_model.pca = pca
	genetic_model.save('genetic_model'+str(identifier))
	return genetic_model


def TrainHebian(identifier, num_epochs=2_000):
	hebian_model = RNN(input_size, hidden_size, output_size)
	#train model using Hebian learning
	trainer = Hebian(hebian_model, task, alpha_trace = 0.5)
	trainer.TrainHebbian(num_trials=num_epochs)
	F = hebian_model.GetF()
	roots, pca = FindFixedPoints(F, [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,\
				-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1])
	hebian_model.pca = pca
	hebian_model.save('hebian_model'+str(identifier))
	return hebian_model


def TrainFORCE(identifier, num_epochs=2_000):
	force_model = RNN(input_size, hidden_size, output_size)
	#train model using FORCE
	trainer = Force(force_model, task, alpha=1000)
	trainer.trainForce(num_trials=num_epochs)
	F = force_model.GetF()
	roots, pca = FindFixedPoints(F, [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,\
				-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1])
	force_model.pca = pca
	force_model.save('force_model'+str(identifier))
	return force_model
