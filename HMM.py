

import random
import argparse
import codecs
import os
import numpy

import pandas as pd
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
        self.rounding_number = 4
        # self.states.remove("#")

    def get_iterable_states(self):
        temp_states = self.states

        if "#" in temp_states:
            temp_states.remove("#")

        return temp_states

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

        transitions['#']['#'] = 0

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

        # observation = ["#"] + observation

        states = self.get_iterable_states()

        num_states = len(states)
        num_obs = len(observation)

        print("The current observation is : ", observation)

        # Initialize the forward matrix
        forward_matrix = numpy.zeros((num_states + 1, num_obs + 1))

        # Initialize the first column of the forward matrix
        forward_matrix[0][0] = 1 # For the start state

        for num_state, curr_state in enumerate(states):
                forward_matrix[num_state + 1][1] = round(self.transitions['#'][curr_state] * self.emissions[curr_state].get(observation[0], 0), self.rounding_number)

        for i in range(2, num_obs + 1):
            # print("The current observation is : ", observation[i-1])
            for curr_state_index in range(1, num_states + 1):
                curr_state = self.states[curr_state_index - 1]
                curr_sum = 0

                for prev_state_index in range(1, num_states + 1):
                    prev_state = self.states[prev_state_index - 1]

                    curr_sum += (self.emissions[curr_state].get(observation[i-1], 0) * \
                                self.transitions[prev_state].get(curr_state, None)* \
                                forward_matrix[prev_state_index][i-1])


                forward_matrix[curr_state_index][i] = curr_sum #round(curr_sum, self.rounding_number)

        return forward_matrix

    def pretty_print_matrix(self, matrix, cols, rows):
        df = pd.DataFrame(matrix, columns=cols, index=rows)
        print(df)

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """

        states = self.get_iterable_states()

        num_states = len(states)
        num_obs = len(observation)

        print("The current observation is : ", observation)

        # Initialize the forward matrix
        forward_matrix = numpy.zeros((num_states + 1, num_obs + 1))
        backpointers = numpy.zeros((num_states + 1, num_obs + 1))

        # Initialize the first column of the forward matrix
        forward_matrix[0][0] = 1

        for num_state, curr_state in enumerate(states):
            forward_matrix[num_state + 1][1] = round(self.transitions['#'][curr_state] * self.emissions[curr_state].get(observation[0], 0), self.rounding_number)

        for i in range(2, num_obs + 1):
            curr_obs = observation[i - 1]

            for curr_state_index in range(1, num_states + 1):
                curr_state = self.states[curr_state_index - 1]
                values = []

                for prev_state_index in range(1, num_states + 1):
                    prev_state = self.states[prev_state_index - 1]

                    values.append(forward_matrix[prev_state_index][i-1] * self.transitions[prev_state].get(curr_state, 0) * self.emissions[curr_state].get(curr_obs, 0))

                val = max(values)
                backpointers[curr_state_index][i] = values.index(val) + 1

        best_list = []
        best = max(forward_matrix[:, num_obs])
        best_index = list(forward_matrix[:, num_obs]).index(best)

        for i in range(num_obs, 0, -1):
            best_list.append(states[best_index])
            best_index = max(backpointers[best_index][i])

        best_list.reverse()

        return best_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="provide the name of the file")
    parser.add_argument("--generate", type=int, help="provide the length of the output sequence", required=False)
    parser.add_argument("--forward", type=str, help="provide the file name of observation", required=False)
    parser.add_argument("--viterbi", type=str, help="provide the file name of observation", required=False)

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
                    states = model.get_iterable_states()
                    model.pretty_print_matrix(model.forward(line), ["#"] + line, ["#"] + states )
                    # break

    if args.viterbi:
        with open(args.viterbi, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    line = line.split()
                    states = model.get_iterable_states()
                    print(model.viterbi(line))
                    # break