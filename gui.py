from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageGrab
import numpy as np
from nn.neural_network import *


class Paint(object):

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.pen)
        self.pen_button.grid(row=0, column=0)

        self.eraser_button = Button(self.root, text='eraser',
                                    command=self.eraser)
        self.eraser_button.grid(row=0, column=1)

        self.canvas_reset_button = Button(self.root, text='reset',
                                          command=self.canvas_reset)
        self.canvas_reset_button.grid(row=0, column=2)

        self.guess_button = Button(self.root, text='guess', command=self.guess)
        self.guess_button.grid(row=0, column=3)

        self.show_box_button = Button(self.root, text="show bounding box",
                                      command=self.draw_box)
        self.show_box_button.grid(row=0, column=4)

        self.guess_label = Label(self.root, text="No Guess Yet")
        self.guess_label.grid(row=0, column=5)

        self.scale_canvas = 20
        self.c = Canvas(self.root, bg='black', width=28*self.scale_canvas,
                        height=28*self.scale_canvas)
        self.c.grid(row=1, columnspan=6)

        self.network = NeuralNetwork([784, 30, 10], None, None)
        self.network.load_network("cross_entropy.pkl")

        self.setup()
        self.root.title("Digit Recognizer")
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.thickness = 10
        self.color = None
        self.eraser = False
        self.active_button = None
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def pen(self):
        self.color = 'white'
        self.activate_button(self.pen_button)

    def eraser(self):
        self.color = 'black'
        self.activate_button(self.eraser_button, eraser_mode=True)

    def canvas_reset(self):
        self.guess_label['text'] = "No Guess Yet"
        self.c.delete('all')
        self.draw_box()

    def draw_box(self):
        first_coord = 4*self.scale_canvas
        second_coord = 4*self.scale_canvas + 20*self.scale_canvas
        self.c.create_rectangle(first_coord, first_coord,
                                second_coord, second_coord, outline="#fb0")

    def guess(self):
        # screenshot and crop to canvas
        x = self.root.winfo_rootx() + self.c.winfo_x()
        y = self.root.winfo_rooty() + self.c.winfo_y()

        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()

        img = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
        img.thumbnail((28, 28))
        img.save("pic.png")
        pix = np.array(img)
        pix = np.reshape(pix, (784, 1))/255.0
        out = self.network.forward_prop(pix)
        print(out)
        self.guess_label['text'] = np.argmax(out)

    def activate_button(self, new_button, eraser_mode=False):
        if self.active_button:
            self.active_button.config(relief=RAISED)

        new_button.config(relief=SUNKEN)
        self.active_button = new_button
        self.eraser = eraser_mode

    def paint(self, event):
        if self.active_button == self.pen_button or self.eraser:

            if self.old_x and self.old_y:
                self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                                   width=self.thickness, fill=self.color,
                                   capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.old_x = event.x
            self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()
