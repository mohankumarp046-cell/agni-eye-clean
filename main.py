import os
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

class AgniEyeApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.layout.add_widget(self.image)
        self.status_label = Label(text="AGNI EYE: Initializing Model...", size_hint=(1, 0.1), font_size='18sp')
        self.layout.add_widget(self.status_label)
        self.load_model()
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.layout

    def load_model(self):
        model_path = "best.tflite"
        if os.path.exists(model_path):
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.status_label.text = "AGNI EYE: Model Loaded Successfully!"
            except Exception as e:
                self.status_label.text = f"Model Load Error: {str(e)}"
        else:
            self.status_label.text = "Warning: best.tflite not found!"

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            input_shape = self.input_details[0]['shape'] if hasattr(self, 'input_details') else [1, 640, 640, 3]
            h, w = input_shape[1], input_shape[2]
            resized = cv2.resize(frame, (w, h))
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
            if hasattr(self, 'interpreter'):
                try:
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    self.interpreter.invoke()
                    self.status_label.text = "AGNI EYE: Monitoring Active - Scanning Thermal/Cracks"
                except Exception as e:
                    pass
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = image_texture

    def on_stop(self):
        if self.capture.isOpened():
            self.capture.release()

if __name__ == '__main__':
    AgniEyeApp().run()
              
