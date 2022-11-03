import numpy as np
import matplotlib.pyplot as plt

def learningcurve_plots(cost_hist: np.array, learning_speed_hist = None, logy: bool = False):
    """
    cost_hist -- history of cost values, as numpy-array of shape (T,1)
    learning_speed_hist -- history of learning speed values, as numpy-array of shape (T,1)
    logy -- if set to True will plot the y axis at logarithmic scale
    """
    plt.figure(1)
    T = len(cost_hist)
    if logy:
        plt.semilogy(np.arange(T),cost_hist,'b-')
    else:
        plt.plot(np.arange(T),cost_hist,'b-')
    plt.title("Cost")
    
    if not learning_speed_hist:
        return None
    plt.figure(2)
    T = len(learning_speed_hist)
    if logy:
        plt.semilogy(np.arange(T),learning_speed_hist,'g-')
    else:
        plt.plot(np.arange(T),learning_speed_hist,'g-')
    plt.title("Learning Speed")