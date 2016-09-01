# bbyy_jet_classifier
This package is used by the ATLAS HH->bbyy search.
It defines a classifier to choose which non-b-tagged jet to pair with the b-tagged jet in 1-tag events.
This code uses the ```skTMVA``` class from ```https://github.com/yuraic/koza4ok``` to convert ```scikit-learn``` output into ROOT TMVA xml input.

# Example: training on SM inputs
./run_classifier.py --input inputs/SM_bkg_photon_jet.root inputs/SM_hh.root --exclude Delta_phi_jb --strategy RootTMVA sklBDT --max_events 1000

# Example: testing previously trained classifier on BSM inputs -- NB. be careful about double counting here
./run_classifier.py --input inputs/X275_hh.root --exclude Delta_phi_jb --strategy sklBDT --ftrain 0 --training_sample SM_merged

./run_classifier.py --input inputs/X300_hh.root --exclude Delta_phi_jb --strategy sklBDT --ftrain 0 --training_sample SM_merged

./run_classifier.py --input inputs/X325_hh.root --exclude Delta_phi_jb --strategy sklBDT --ftrain 0 --training_sample SM_merged

./run_classifier.py --input inputs/X350_hh.root --exclude Delta_phi_jb --strategy sklBDT --ftrain 0 --training_sample SM_merged

./run_classifier.py --input inputs/X400_hh.root --exclude Delta_phi_jb --strategy sklBDT --ftrain 0 --training_sample SM_merged

# Example: training and testing on all inputs
./run_classifier.py --input inputs/*root --exclude Delta_phi_jb

# Example: evaluate event-level performance
./evaluate_event_performance.py X275_hh X300_hh X325_hh X400_hh SM_bkg_photon_jet SM_hh

# Inputs
The most recent set of input TTrees are in:
/afs/cern.ch/user/j/jrobinso/work/public/ML_inputs
