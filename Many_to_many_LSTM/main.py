from Neuralnet import Neuralnet
from Data import Data
from Ect import Ect


def main():
    '''
    main function for the programm...

    Returns
    -------
    None.

    '''
    Ect.create_folder()
    Ect.gen_sin()
    Data.prepare()
    Data.plot(0, 0)
    Neuralnet.define()
    
    i = 0
    while True:
        i += 1
        
        Neuralnet.train(Data.x_train, Data.y_train)
        y_pred = Neuralnet.feedforward(Data.seq_lenght)
        Data.plot(y_pred[0], i)

        if input('Continue learning? [y]/[N]: ') != 'y': break

    Ect.print_legend()

main()
