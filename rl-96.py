belief_intent = 0.5

for human_signal in [1,0,1]:
    belief_intent = 0.6*belief_intent + 0.4*human_signal
    print("Belief about intent:", belief_intent)
