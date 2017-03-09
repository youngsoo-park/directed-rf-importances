import sklearn
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MonteCarlo(object):


    def __init__(self,model,X_in,X_names):


        self.model = model
        self.X = X_in
        self.feat_names = X_names
        self.feat_min = np.amin(X_in,axis=0)
        self.feat_max = np.amax(X_in,axis=0)
        self.n_feat = len(self.feat_max)
    

    def generate(self,n_sample,n_bins):

        '''
        Generates a Monte Carlo sample of the feature space
        and the derived conditional probabilities P(y|x_i) for each feature x_i
        '''

        # Generate Monte Carlo samples

        sample_X = np.array([[random.uniform(self.feat_min[i],self.feat_max[i]) for i in range(self.n_feat)] for _ in range(n_sample)])
        sample_y = self.model.predict_proba(sample_X)[:,1]
        sample = np.c_[sample_X,sample_y]


        # Generate conditional probability curves or "risk score curves"

        curves = []

        for i in range(self.n_feat):

            ssize = (self.feat_max[i] - self.feat_min[i])/n_bins
            probs = []

            for j in range(n_bins):

                subsample = sample[ (sample[:,i] > self.feat_min[i] + j*ssize) & (sample[:,i] < self.feat_min[i] + (j+1)*ssize) , : ]
                probs.append([np.sum(subsample[:,-1]) / len(subsample[:,-1]), np.std(subsample[:,-1])])

            curves.append(probs)

        self.curves = np.array(curves)
        self.sample = sample
        self.n_bins = n_bins



    def plot_curves(self):

        '''
        Plots the genereated conditional probability curves
        '''

        for i in range(self.n_feat):

            curve = self.curves[i,:,:]

            vals = curve[:,0]
            sigmas = curve[:,1]

            plt.figure(figsize=(8,6))
            plt.title(self.feat_names[i])

            ssize = (self.feat_max[i] - self.feat_min[i]) / self.n_bins
            feat_range = np.arange(self.feat_min[i]+0.5*ssize, self.feat_max[i], ssize)

            plt.errorbar(feat_range, vals, sigmas)
            plt.show()
            plt.clf()




    def edge_detect(self):
  
        '''
        Detects sharp transitions in the conditional probability function,
        returns their locations and magnitudes,
        and plots the curves and transitions
        '''

        # Clean the generated curves for non-numerical and single-valued features

        clean_curves = []

        for i in range(self.n_feat):

            if math.isnan(self.curves[i,0,0]): 
                #print(model['feature_importances_names'][i])
                #print(feat_min[i],feat_max[i])
                continue
            else:
                clean_curves.append(i)


        # Perform edge detection, print and plot transitions

        transitions_all = []

        for i_curve in clean_curves:

            curve = self.curves[i_curve,:,:]

            vals = curve[:,0]
            sigmas = curve[:,1]

            subarr = []
            subarr.append(vals[0])
            transitions = []
            
            for i in range(self.n_bins-1):

                mean = np.mean(subarr)
                subarr.append(vals[i+1])
                std = np.std(subarr)
                
                #if abs(vals[i+1]-mean) > 2.*std:
                if abs(vals[i+1]-mean) > 0.1 * sigmas[i+1]:

                    #print("Edge detected after bin ", i, ", magnitude ", vals[i+1]-mean)
                    subarr = []
                    subarr.append(vals[i+1])
                    transitions.append([i,vals[i+1]-mean])
                    
            transitions_all.append(transitions)

            plt.figure(figsize=(8,6))
            plt.title(self.feat_names[i_curve])

            ssize = (self.feat_max[i_curve] - self.feat_min[i_curve]) / self.n_bins
            feat_range = np.arange(self.feat_min[i_curve]+0.5*ssize, self.feat_max[i_curve], ssize)

            plt.errorbar(feat_range, vals, sigmas)
            
            for i in range(len(transitions)):

                xpos = transitions[i][0]
                plt.arrow( (feat_range[xpos] + feat_range[xpos+1]) / 2., (vals[xpos] + vals[xpos+1]) / 2., 0, transitions[i][1], head_length=ssize/10., head_width=ssize/1.2, width=ssize/20., color='red')
            
            #plt.savefig(model['feature_importances_names'][xx]+'.png')
            plt.show()
            plt.clf()

        self.transitions = transitions_all





