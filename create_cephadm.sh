#!/bin/bash
#设定普通用户
useradd -m -s /bin/bash cephadm
echo cephadm:123456 | chpasswd
echo "cephadm ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/cephadm
chmod 0440 /etc/sudoers.d/cephadm
