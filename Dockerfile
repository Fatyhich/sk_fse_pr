FROM  ubuntu:24.04

RUN apt update && apt install -y bash  make git curl ca-certificates build-essential gcc vim python3 python3-pip python3-venv libgl1 libglib2.0-0


# Добавляем пользователя
RUN adduser --disabled-password --gecos '' captain \
    && adduser captain sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Создаем необходимые папки
RUN mkdir /mnt/code && chmod a+rwx /mnt/code && \
    mkdir /mnt/data && chmod a+rwx /mnt/data

# Устанавливаем рабочую директорию
WORKDIR /mnt/code

# Устанавливаем необходимые пакеты
COPY Makefile_docker /mnt/code/

RUN make -f /mnt/code/Makefile_docker prereqs

COPY apply_masks.pyx setup.py  Makefile /mnt/code/

COPY src/ /mnt/code/src

RUN make -f /mnt/code/Makefile_docker build

COPY test/ /mnt/code/test

RUN make -f /mnt/code/Makefile_docker test