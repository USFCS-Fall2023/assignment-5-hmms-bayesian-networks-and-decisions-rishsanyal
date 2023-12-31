

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
        self.rounding_number = 10
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
                forward_matrix[num_state + 1][1] = self.transitions['#'][curr_state] * self.emissions[curr_state].get(observation[0], 0)

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

        result = self.__get_forward_result(forward_matrix)

        return result

    def __get_forward_result(self, forward_matrix):
        last_max_observation_index = numpy.argmax(forward_matrix[:, -1])
        last_max_observation = self.states[last_max_observation_index - 1]
        return last_max_observation

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
            forward_matrix[num_state + 1][1] = self.transitions['#'][curr_state] * self.emissions[curr_state].get(observation[0], 0)
            backpointers[num_state + 1][1] = 0

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

            for curr_state_index in range(1, num_states + 1):

                curr_state = self.states[curr_state_index - 1]

                max_val = 0
                max_index = 0

                for prev_state_index in range(1, num_states + 1):
                    prev_state = self.states[prev_state_index - 1]

                    curr_val = (self.emissions[curr_state].get(observation[i-1], 0) * \
                                self.transitions[prev_state].get(curr_state, None)* \
                                forward_matrix[prev_state_index][i-1])

                    if curr_val > max_val:
                        max_val = curr_val
                        max_index = prev_state_index

                backpointers[curr_state_index][i] = max_index

        return self.__get_viterbi_path(backpointers, observation, forward_matrix)

    def __get_viterbi_path(self, backpointers, observation, forward_matrix):
        states = self.get_iterable_states()
        num_obs = len(observation)

        most_likely_path = []

        forward_df = pd.DataFrame(forward_matrix, columns=["#"] + observation, index=["#"] + states)
        backpointer_df = pd.DataFrame(backpointers, columns=["#"] + observation, index=["#"] + states)

        last_observation = observation[-1]

        last_max_observation_index = forward_df.idxmax()

        most_likely_path.append(last_max_observation_index[last_observation])

        most_likely_path_index_list = []

        most_likely_path_index = backpointer_df.axes[0].get_loc(last_max_observation_index[last_observation])

        most_likely_path_index_list.append(most_likely_path_index)

        for i in range(num_obs, -1, -1):
            most_likely_path_index = int(backpointer_df.iloc[most_likely_path_index][i])
            most_likely_path_index_list.append(most_likely_path_index)

        temp_col = ["#"] + states

        for i in most_likely_path_index_list[::-1]:
            most_likely_path.append(temp_col[i])

        return most_likely_path[3:]



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
                    print("Final state is : ", model.forward(line))

    if args.viterbi:
        with open(args.viterbi, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    line = line.split()
                    print(model.viterbi(line))