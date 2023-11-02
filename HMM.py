

import random
import argparse
import codecs
import os
import numpy

from pprint import pprint

import sys

from collections import defaultdict

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

        self.states = list(self.transitions.keys())
        self.states.remove("#")

    ## part 1 - you do this. - Done
    @staticmethod
    def load(basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        transitions = defaultdict(dict)
        emissions = defaultdict(dict)

        with open(basename + ".emit", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    line = line.split()
                    emissions[line[0]].update({line[1]: float(line[2])})


        with open(basename + ".trans", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    line = line.split()
                    transitions[line[0]].update({line[1]: float(line[2])})


        return HMM(transitions, emissions)



   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""

        initial_state = self.transitions["#"]

        states = []
        observations = []

        state = numpy.random.choice(list(initial_state.keys()), p=list(initial_state.values()))

        states.append(state)

        for i in range(n-1):
            initial_state = self.transitions[state]
            state = numpy.random.choice(list(initial_state.keys()), p=list(initial_state.values()))

            states.append(state)

        for state in states:
            initial_state = self.emissions[state]
            observation = numpy.random.choice(list(initial_state.keys()), p=list(initial_state.values()))

            observations.append(observation)

        return observations, states


    def forward(self, observation):
        ## Implement forward algorithm here.

        num_states = len(self.states)
        num_obs = len(observation)

        print("The current observation is : ", observation)

        # Initialize the forward matrix
        forward_matrix = numpy.zeros((num_states, num_obs))

        for i, curr_state in enumerate(self.states):
            forward_matrix[i][0] = self.transitions['#'][curr_state] * self.emissions[curr_state].get(observation[0], 0)

        for i in range(1, num_obs):
            for state_index, curr_state in enumerate(self.states):
                curr_sum = 0
                for j, prev_state in enumerate(self.states):
                    curr_sum += forward_matrix[j][i-1] * self.transitions[prev_state][curr_state] * self.emissions[curr_state].get(observation[i], 0)
                forward_matrix[state_index][i] = curr_sum


        return forward_matrix

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="provide the name of the file")
    parser.add_argument("--generate", type=int, help="provide the length of the output sequence", required=False)
    parser.add_argument("--forward", type=str, help="provide the file name of observation", required=False)


    parser.add_argument("--transitions", help="provide the file name of observation", action='store_true')
    parser.add_argument("--emissions", help="provide the file name of observation", action='store_true')

    args = parser.parse_args()

    model = HMM.load(args.filename)

    if args.transitions:
        pprint(model.transitions)
    if args.emissions:
        pprint(model.emissions)

    if args.transitions or args.emissions:
        sys.exit(0)

    if args.generate:
        a, b = model.generate(args.generate)

        print("-"*50)
        print(a)
        print("\n")
        print(b)
        print("-"*50)

    if args.forward:
        with open(args.forward, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    line = line.split()
                    print(model.forward(line))
                    break