FROM python:3.6

RUN mkdir -p /root/denoising_project
WORKDIR /root/denoising_project

COPY requirements.txt /root/denoising_project

# all
RUN pip install -r requirements.txt

# torch
RUN pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html


# Копируем код из локального контекста в рабочую директорию образа
COPY . .
# Настраиваем команду, которая должна быть запущена в контейнере во время его выполнения
ENTRYPOINT ["python", "denoiser.py", "/dataset", "/results", "--device_name", "cpu"]

