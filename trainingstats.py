def get_training_stats(mlp, dset, nepochs, batch_size):
        
    # Declaration
    train_set_temp, val_set, test_set = dset
    train_X_temp, train_Y_temp = train_set_temp
    val_X, val_labels = val_set
    test_X, test_labels = test_set
    indexes = np.arange(train_X_temp.shape[0])
    train_loss = np.zeros(nepochs)
    train_error = np.zeros(nepochs)
    val_loss = np.zeros(nepochs)
    val_error = np.zeros(nepochs)
    confusion_matrix = np.zeros((10,10))
    
    for e in range(nepochs):
        np.random.shuffle(indexes)
        train_X = train_X_temp[indexes]
        train_labels = train_Y_temp[indexes]
        tloss = 0
        vloss = 0
        tcorrect = 0
        vcorrect = 0
        
        # Train
        for i in range(int(train_X.shape[0]/batch_size)):
            mlp.zero_grads()
            toutput = mlp.forward(train_X[batch_size*i:batch_size*(i+1)])
            tlabels = train_labels[batch_size*i:batch_size*(i+1)]
            mlp.backward(tlabels)
            loss = []
            max_elements = np.amax(toutput, axis = 1).reshape((toutput.shape[0], 1))
            sm = np.exp(toutput-max_elements)/np.sum(np.exp(toutput-max_elements), axis = 1).reshape((toutput.shape[0], 1))
            for element in -np.sum(tlabels*np.log(sm), axis = 1):
                 loss.append(element)
            tloss += sum(loss)
            mlp.step()
            for j in range(toutput.shape[0]):
                if np.argmax(toutput[j]) == np.argmax(tlabels[j]):
                    tcorrect += 1
            
        # Validation
        for i in range(int(val_X.shape[0]/batch_size)): 
            mlp.zero_grads()
            vlabels = val_labels[batch_size*i:batch_size*(i+1)]
            voutput = mlp.forward(val_X[batch_size*i:batch_size*(i+1)])
            loss = []
            max_elements = np.amax(voutput, axis = 1).reshape((voutput.shape[0], 1))
            sm = np.exp(voutput-max_elements)/np.sum(np.exp(voutput-max_elements), axis = 1).reshape((voutput.shape[0], 1))
            for element in -np.sum(vlabels*np.log(sm), axis = 1):
                loss.append(element)
            vloss += sum(loss)
            for j in range(voutput.shape[0]):
                if np.argmax(voutput[j]) == np.argmax(vlabels[j]):
                    vcorrect += 1

        train_loss[e] = tloss / train_X.shape[0]
        train_error[e] = 1.0 - (tcorrect / val_X.shape[0])
        val_loss[e] = vloss / val_X.shape[0]
        val_error[e] = 1.0 - (vcorrect / val_X.shape[0])
        
    for i in range(int(test_X.shape[0]/batch_size)):
        mlp.zero_grads()
        output = mlp.forward(test_X[batch_size*i:batch_size*(i+1)])
        pred = np.argmax(output, axis = 1)
        actual = np.argmax(test_labels[batch_size*i:batch_size*(i+1)], axis = 1 )
        for j in range(batch_size):
             confusion_matrix[pred[j]][actual[j]]+=1
             
    return train_loss, train_error, val_loss, val_error, confusion_matrix
