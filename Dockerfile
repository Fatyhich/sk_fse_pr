FROM  ubuntu:24.04

RUN apt update && apt install -y bash  make git curl ca-certificates build-essential gcc vim python3 python3-pip python3-venv libgl1 libglib2.0-0


# Добавляем пользователя
RUN adduser --disabled-password --gecos '' captain \
    && adduser captain sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Устанавливаем необходимые пакеты
RUN pip install pandas fire tqdm matplotlib torch segment-anything torchvision opencv-python six --break-system-packages

RUN pip install --upgrade six --break-system-packages && pip install --force-reinstall six --break-system-packages

# Создаем необходимые папки
RUN mkdir /mnt/code && chmod a+rwx /mnt/code && \
    mkdir /mnt/data && chmod a+rwx /mnt/data

# Устанавливаем рабочую директорию
WORKDIR /mnt/code