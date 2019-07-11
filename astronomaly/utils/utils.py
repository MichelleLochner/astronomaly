import matplotlib.pyplot as plt

class ImageCycler:
    def __init__(self, images):
        """
        Convenience object to cycle through a list of images inside a jupyter notebook.

        Parameters
        ----------
        images : list
            List of numpy arrays to display as images
        """

        self.current_ind = 0
        self.images = images


    def onkeypress(self, event):
        """
        Matplotlib event handler for left and right arrows to cycle through images.

        Parameters
        ----------
        event

        Returns
        -------

        """
        plt.gcf()
        if event.key=='right' and self.current_ind<len(self.images):
            self.current_ind += 1

        elif event.key=='left' and self.current_ind>0:
            self.current_ind -= 1

        plt.clf()
        event.canvas.figure.gca().imshow(self.images[self.current_ind], origin='lower', cmap='hot')

        plt.title(self.current_ind)
        event.canvas.draw()


    def cycle(self):
        """
        Creates the plots and binds the event handler
        """

        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        plt.imshow(self.images[self.current_ind], origin='lower',cmap='hot')
        plt.title(self.current_ind)