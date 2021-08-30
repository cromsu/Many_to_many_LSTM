import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Data:
    '''
    Data class. Contains methods to handle the data for the neural network.
        
    Methods
    -------
    prepare
    plot
    normalize
    denormalize
    
    '''
    def prepare():
        '''
        Reads 'sequense.dat' file, creates training data.:
            Data.y_train
            Data.x_train


        Returns
        -------
        None.

        '''
        y_train = np.loadtxt('./sequence.dat')
        
        Data.b = np.linalg.norm(y_train)
        seq_lenght = y_train.shape[0]
        
        x_train = np.linspace(0,seq_lenght-1, seq_lenght)
        x_train = x_train.reshape(1,len(x_train),1)
        
        y_train = Data.normalize(y_train)
        y_train = y_train.reshape(1,len(y_train),1)
        
        Data.y_train = y_train
        Data.x_train = x_train
        Data.seq_lenght = seq_lenght

        return None
     
    def plot(y_model, i):
        '''
        Writes plots to "./plots".

        Parameters
        ----------
        y_model : ndarray
            models prediction.
        i : int
            for file name.

        Returns
        -------
        None.

        '''
        img_file = './plots/diagramm(' + str(i) + ').png'
        y_real = Data.denormalize(Data.y_train[0])
        y_model = Data.denormalize(y_model)

        mpl.use('Agg')
        plt.clf()
        plt.plot(y_real, label='data', color='red')
        plt.plot(y_model, label='prediction',
                 linewidth = 3,
                 linestyle = (0,(0.1,2)),
                 dash_capstyle = 'round',
                 color = 'green')
        plt.grid()
        plt.title(('learningloop '+str(i)))
        plt.savefig(img_file, dpi=1000)
        
        return None
        
    def denormalize(array):
        '''
        Denormalizes an array with the class varialbe Data.b.

        Parameters
        ----------
        array : array, float
            array to denormalize.

        Returns
        -------
        array : array, float
            denormalized array.

        '''
        return array * Data.b

    def normalize(array):
        '''
        Normalizes an array. Creates normalizetion constant as class variable b.

        Parameters
        ----------
        array : array, float
            Array to normalize.

        Returns
        -------
        norm : array, float
            Normalized array.

        '''
        norm = array / Data.b
        return norm
