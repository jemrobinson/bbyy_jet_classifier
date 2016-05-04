#! /bin/bash
echo "Running autopep8"
autopep8 bbyy_jet_classifier --recursive --ignore=E501 --in-place
autopep8 viz --recursive --ignore=E501 --in-place
for pyfile in *py; do autopep8 $pyfile --recursive --ignore=E501 --in-place; done
echo "Running pylint"
pylint bbyy_jet_classifier
pylint viz
for pyfile in *py; do pylint $pyfile; done
