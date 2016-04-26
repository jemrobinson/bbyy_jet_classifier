# bbyy_jet_classifier
Classifier to determine which jet-pairs to use for analysis

# Example: training on SM inputs
./run_classifier.py --input SM_merged.root --exclude MV2c20_FCBE_70 MV2c20_FCBE_77 MV2c20_FCBE_85 --strategy sklBDT

# Example: testing on BSM inputs
./run_classifier.py --input X275_hh.root --exclude MV2c20_FCBE_70 MV2c20_FCBE_77 MV2c20_FCBE_85 --strategy sklBDT --ftrain 0

The most recent set of input TTrees are in:
/afs/cern.ch/user/j/jrobinso/work/public/ML_inputs
