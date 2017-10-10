def get_pij_using_loop():
    
    """
    Gets Pij by looping through patients. Is appropriate when the 
    non-loop version causes memory errors.            
    """
    
    with tf.name_scope("pij_loop"):
        
        n = tf.shape(self.X_transformed)[0]
        n = tf.cast(n, tf.int32)
        #n = self.X_transformed.get_shape().as_list()[0]
        #n = tf.cast(tf.size(self.T)-1, tf.int32)
        
        # first patient
        #sID = 0
        #patient = self.X_transformed[sID, :]
        #patient_normax = (patient[None, :] - self.X_transformed)**2
        #normAX = tf.reduce_sum(patient_normax, axis=1)
        #normAX = normAX[None, :]
        normAX = tf.Variable(tf.zeros([n, n]), validate_shape=False)
        
        # all other patients
        def _append_normAX(sID, normAX):
        
            """append normAX for a single patient to existing normAX"""
            
            # calulate normAX for this patient    
            patient = self.X_transformed[sID, :]
            patient_normax = (patient[None, :] - self.X_transformed)**2
            patient_normax = tf.reduce_sum(patient_normax, axis=1)
            patient_normax = patient_normax[None, :]
        
            # append to existing list
            #normAX = tf.concat((normAX, patient_normax[None, :]), axis=0)
            #a = normAX[sID, :]
            #normAX = tf.assign(normAX[sID, :], patient_normax[None, :])
            normAX = normAX[sID, :].assign(patient_normax[None, :])
            #normAX = tf.Variable(normAX)
            
            # sID++
            sID = tf.cast(tf.add(sID, 1), tf.int32)
            
            return sID, normAX
            
        
        
        
        # Go through all patients and add normAX
        #sID = tf.cast(tf.Variable(1), tf.int32)
        sID = tf.cast(tf.Variable(0), tf.int32)
        
        c = lambda sID, normAX: tf.less(sID, tf.cast(n, tf.int32))
        b = lambda sID, normAX: _append_normAX(sID, normAX)
        
        (sID, normAX) = tf.while_loop(c, b, 
                        loop_vars = [sID, normAX])#, 
                        #shape_invariants = 
                        #[sID.get_shape(), tf.TensorShape([None, n])])
                    
    return normAX