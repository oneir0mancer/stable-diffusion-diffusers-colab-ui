from ipycanvas import MultiCanvas, hold_canvas
from ipywidgets import ColorPicker, IntSlider, link, AppLayout, HBox, VBox, Button, Checkbox
from ipywidgets import Image

class DrawContext:
    def __init__(self):
        self.drawing = False
        self.position = None
        self.shape = []

class CanvasHolder:
    def __init__(self, width=512, height=512, line_color="#749cb8"):
        self.multicanvas = MultiCanvas(2, width=width, height=height)
        self.canvas = self.multicanvas[1]   #foreground
        self.background = self.multicanvas[0]
        self.context = DrawContext()
        self.save_file = "temp.png"

        self.canvas.on_mouse_down(self.on_mouse_down)
        self.canvas.on_mouse_move(self.on_mouse_move)
        self.canvas.on_mouse_up(self.on_mouse_up)

        self.picker = ColorPicker(description="Color:", value=line_color)
        link((self.picker, "value"), (self.canvas, "stroke_style"))
        link((self.picker, "value"), (self.canvas, "fill_style"))
        self.canvas.stroke_style = line_color

        self.line_width_slider = IntSlider(1, min=1, max=20, description="Line Width:")
        link((self.line_width_slider, "value"), (self.canvas, "line_width"))

        self.fill_check = Checkbox(value=False, description="Fill shape")

        self.clear_btn = Button(description="Clear")
        def clear_handler(b):
            self.canvas.sync_image_data = False
            with hold_canvas():
                self.canvas.clear()
        self.clear_btn.on_click(clear_handler)

        def save_to_file(*args, **kwargs):
            self.canvas.to_file(self.save_file)
        self.canvas.observe(save_to_file, "image_data")

    def on_mouse_down(self, x, y):
        self.canvas.sync_image_data = False
        self.context.drawing = True
        self.context.position = (x, y)
        self.context.shape = [self.context.position]

    def on_mouse_move(self, x, y):
        if not self.context.drawing: return
        self.context.shape.append((x, y))
        with hold_canvas():
            self.canvas.stroke_line(self.context.position[0], self.context.position[1], x, y)
            if self.canvas.line_width > 3: self.canvas.fill_circle(x, y, self.canvas.line_width)
            self.context.position = (x, y)

    def on_mouse_up(self, x, y):
        self.context.drawing = False
        with hold_canvas():
            self.canvas.stroke_line(self.context.position[0], self.context.position[1], x, y)
            if self.fill_check.value: self.canvas.fill_polygon(self.context.shape)
        self.context.shape = []

    def add_background(self, filepath, resize = False):
        with hold_canvas():
            self.background.clear()
            if resize:
                self.background.draw_image(Image.from_file(filepath), 0,0, width=300, height=300)
            else:
                self.background.draw_image(Image.from_file(filepath))

    def save_mask(self, filepath = "temp.png"):
        self.save_file = filepath
        self.canvas.sync_image_data = True  #optimization hack to not sync image all the time

    @property
    def render_element(self): 
        return HBox([self.multicanvas, 
                     VBox([self.picker, self.line_width_slider, self.fill_check, self.clear_btn])])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()