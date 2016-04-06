from collections import OrderedDict
import ROOT

class root2numpy(object) :

  @classmethod
  def get_tree_variables( cls, input_tree, excluded_variables=[] ) :
    variable_dict = OrderedDict()
    type2char = { "Int_t":"I", "Double_t":"D", "Float_t":"F" }
    for leaf in sorted(input_tree.GetListOfLeaves()) :
      variable_name = leaf.GetName()
      if variable_name not in excluded_variables :
        variable_dict[variable_name] = type2char[leaf.GetTypeName()]
    return variable_dict

  @classmethod
  def tree2array( cls, input_tree, excluded_variables=[] ) :
    variable_dict = cls.get_tree_variables( input_tree, excluded_variables )
    print "not yet functional"
