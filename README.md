# bbyy_jet_classifier
Classifier to determine which jet-pairs to use for analysis

# Example: training on SM inputs
./run_classifier.py --input SM_merged.root --exclude Delta_phi_jb --strategy sklBDT

# Example: testing on BSM inputs -- NB. be careful about double counting here
./run_classifier.py --input X275_hh.root --exclude Delta_phi_jb --strategy sklBDT --ftrain 0

# Inputs
The most recent set of input TTrees are in:
/afs/cern.ch/user/j/jrobinso/work/public/ML_inputs
