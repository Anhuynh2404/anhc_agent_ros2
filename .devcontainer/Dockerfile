FROM althack/ros2:humble-full

ENV DEBIAN_FRONTEND=noninteractive

# ── Hệ thống & ROS/Gazebo packages ─────────────────────────────────────────
RUN apt-get update \
   && apt-get -y install --no-install-recommends \
        ros-humble-gazebo-ros-pkgs \
        ros-humble-gazebo-ros \
        ros-humble-gazebo-plugins \
        python3-pip \
        python3-rosdep \
   #
   # Clean up
   && apt-get autoremove -y \
   && apt-get clean -y \
   && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=dialog

# ── PyTorch với CUDA 11.8 ───────────────────────────────────────────────────
# Cài một lần vào image, không cần cài lại mỗi lần chạy container
RUN pip3 install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu118

# ── Python dependencies của drl_agent ───────────────────────────────────────
# Tất cả packages được scan từ scripts/ để không cần pip install thủ công
RUN pip3 install --no-cache-dir \
        squaternion \
        numpy \
        matplotlib \
        matplotlib-scalebar \
        PyYAML \
        tensorboard

# ── Auto-source workspace khi mở terminal ──────────────────────────────────
ARG WORKSPACE
RUN echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> /home/ros/.bashrc
RUN echo "if [ -f /usr/share/gazebo/setup.bash ]; then source /usr/share/gazebo/setup.bash; fi" >> /home/ros/.bashrc
