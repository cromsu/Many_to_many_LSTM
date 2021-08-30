import shutil
import os
import numpy as np

class Ect:
    '''
    Contains methos for creating a "./plots" folder ,console printing the legend,
    
    for Data.plot() plots.
        
    Methods
    -------
    create_folder
    print_legend
    gen_sin
    
    '''
    def create_folder():
        '''
        Makes shure the "./plots" folder is there.

        Returns
        -------
        None.

        '''
        try:
            shutil.rmtree('./plots')
        except OSError:
            pass
        os.mkdir('./plots')

    def print_legend():
        '''
        Prints the legend to Data.plot() plots created in main funktion.

        Returns
        -------
        None.

        '''
        print('\n')
        print('Done! Plots are saved as a png file in "./plots".')
        print('>>> red   - real data')
        print('>>> green - ai prediction')
        print('* \"diagramm(0).png\" is only the funktion without ai.')
    
    def gen_sin():
        '''
        Generates Sinsequence and writes it to "./sequence.dat" if "sequence.dat"
        is not there. Gets coeffizients by console promts.

        Returns
        -------
        None.

        '''
        if os.path.exists('./sequence.dat') == True: return()
        else: print('There is no data.dat file. Let\'s create one.')

        numbers = int(input('Lenght of sin-sequence y=a*sin(x*b). (int; x>1):'))
        a = float(input('Coefficient a (float):'))
        b = float(input('Coefficient b (float):'))
    
        data = np.empty(numbers)
        for x in range(numbers):
            data[x] = a * np.sin(x*b)

        np.savetxt('./sequence.dat', data)
        
    
