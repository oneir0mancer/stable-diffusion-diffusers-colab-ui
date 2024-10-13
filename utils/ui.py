from ipywidgets import Button, Layout

class SwitchButton:
    def __init__(self, description_on_switch, description_off_switch, layout, 
                 callback_on, callback_off, start_on=False):
        self.description_on_switch = description_on_switch
        self.description_off_switch = description_off_switch
        self.callback_on = callback_on
        self.callback_off = callback_off
        self.state = start_on

        self.button = Button(description="", layout=layout)
        if start_on: self.__set_on_state()
        else: self.__set_off_state()

    def switch_on(self, b):
        if self.state is True: return
        self.callback_on()
        self.__set_on_state()

    def switch_off(self, b):
        if self.state is False: return
        self.callback_off()
        self.__set_off_state()
    
    def __set_on_state(self):
        self.button.on_click(self.switch_on, True)
        self.button.on_click(self.switch_off)
        self.button.description = self.description_off_switch
        self.button.button_style = "success"
        self.state = True

    def __set_off_state(self):
        self.button.on_click(self.switch_off, True)
        self.button.on_click(self.switch_on)
        self.button.description = self.description_on_switch
        self.button.button_style = "danger"
        self.state = False

    @property
    def render_element(self): 
        return self.button

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
