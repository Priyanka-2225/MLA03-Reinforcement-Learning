import numpy as np

patients = np.random.poisson(20, 30)
beds = 25

for p in patients:
    print("Patients:", p, "Beds used:", min(p, beds))
