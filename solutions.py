from HMM import *
from carnet import *
from alarm import *

def forward_solution():
    """Forward algorithm solution."""
    filename = "partofspeech.browntags.trained"
    forward_model = HMM.load(filename)

    forward_filename = "ambiguous_sents.obs"

    with open(forward_filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                line = line.split()
                # states = forward_model.get_iterable_states()
                print("Final state is : ", forward_model.forward(line))

def viterbi_solution():
    """Viterbi algorithm solution."""
    filename = "partofspeech.browntags.trained"
    viterbi_model = HMM.load(filename)

    viterbi_filename = "ambiguous_sents.obs"

    with open(viterbi_filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                line = line.split()
                # states = forward_model.get_iterable_states()
                print("Final state is : ", viterbi_model.viterbi(line))





if __name__ == '__main__':
    forward_solution()
    print("-"*30)
    viterbi_solution()
    print("-"*30)
    print("-"*30)

    #  Given that the car will not move, what is the probability that the battery is not working?
    q = car_infer.query(variables=["Battery"],evidence={"Moves":"no"})
    print(q)

    # Given that the radio is not working, what is the probability that the car will not start?
    a = car_infer.query(variables=["Starts"],evidence={"Radio":"Doesn't turn on"})
    print(a)

    # Given that the battery is working, does the probability of the radio working change if
    # we discover that the car has gas in it?

    b = car_infer.query(variables=["Radio"],evidence={"Battery":"Works", "Gas":"Full"})
    print(b)

    # Given that the car doesn't move, how does the probability of the ignition failing change if we observe that the car does not have gas in it?
    c = car_infer.query(variables=["Ignition"],evidence={"Moves":"no", "Gas":"Empty"})
    print(c)

    # What is the probability that the car starts if the radio works and it has gas in it?
    d = car_infer.query(variables=["Starts"],evidence={"Radio":"turns on", "Gas":"Full"})
    print(d)

    print("-"*30)

    # the probability of Mary Calling given that John called
    a = alarm_infer.query(variables=["MaryCalls"],evidence={"JohnCalls":"yes"})
    print(a)

    # The probability of both John and Mary calling given Alarm
    b = alarm_infer.query(variables=["JohnCalls","MaryCalls"],evidence={"Alarm":"yes"})
    print(b)

    # the probability of Alarm, given that Mary called.
    c = alarm_infer.query(variables=["Alarm"],evidence={"MaryCalls":"yes"})
    print(c)
