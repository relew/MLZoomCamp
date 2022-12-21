FROM svizor42/zoomcamp-dino-dragon-lambda:v2

COPY hw9_notebook.py .

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl
RUN pip install requests
RUN pip install numpy
RUN pip install Pillow

CMD [ "hw9_notebook.lambda_handler" ]