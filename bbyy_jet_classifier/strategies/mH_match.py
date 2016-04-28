from . import BaseStrategy
import logging
import numpy as np
import numpy.lib.recfunctions as rfn

class mHmatch(BaseStrategy) :
  default_output_location = "output/mHmatch"
  classifier_range = ( 0.0, 1.0 )


  def train_and_test( self, training_fraction ) :
    # -- Initialise testing
    logging.getLogger("mHmatch::Train").info( "No training step needed, testing on all events..." )
    self.test_only()

    # -- Construct test events
    logging.getLogger("mHmatch::Train").info( "Combining record arrays..." )
    self.test_events = np.hstack(( self.test_correct_events, self.test_incorrect_events ))



  def test_only( self ) :
    # -- Calculate classifier output and add to trees
    output_arrays = []
    logging.getLogger("mHmatch::Test").info( "Calculating classifier output..." )
    for idx_array, _array in enumerate( [ self.correct_array, self.incorrect_array ] ) :
      output_arrays.append( rfn.append_fields( _array, ["mHmatch","classID"], [_array.idx_by_mH == 0, np.full(_array.size,idx_array,int) ], dtypes=int, asrecarray=True ) )

    # -- Construct record arrays of correct/incorrect events
    logging.getLogger("mHmatch::Test").info( "Constructing record arrays of correct/incorrect events..." )
    self.test_correct_events, self.test_incorrect_events = output_arrays
    self.test_correct_events.dtype.names = [ x.replace("event_weight","weight") for x in self.test_correct_events.dtype.names ]
    self.test_incorrect_events.dtype.names = [ x.replace("event_weight","weight") for x in self.test_incorrect_events.dtype.names ]
