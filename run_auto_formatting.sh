#! /bin/bash
echo "Running autopep8"
for pyfile in *py; do autopep8 $pyfile --recursive --ignore=E501 --in-place; done
autopep8 viz --recursive --ignore=E501 --in-place
autopep8 bbyy_jet_classifier --recursive --ignore=E501 --in-place
echo "Running pylint"
for pyfile in *py; do pylint $pyfile; done
pylint viz
pylint bbyy_jet_classifier
