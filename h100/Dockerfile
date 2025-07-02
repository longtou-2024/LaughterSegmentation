# longtou.2024
FROM gcr.io/deeplearning-platform-release/pytorch-cu121.2-3.py310

RUN apt update
WORKDIR /root
ADD LaughterSegmentation /root/LaughterSegmentation
WORKDIR /root/LaughterSegmentation
RUN ./setup_miniforge.sh forge laughter 3.10
RUN  . ./activate_python.sh  && pip install -r requirements.txt

CMD ["/bin/bash"]
