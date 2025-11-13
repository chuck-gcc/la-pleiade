#!/bin/bash

VM_NAME="test"
VM_ISO="/var/lib/libvirt/images/ubuntu-24.04.3-desktop-amd64.iso"
VM_SEED= "/home/cc/gpu_lab/pleiades/electra/autoinstall/seed.iso"
VM_URL="http://ftp.us.debian.org/debian/dists/stable/main/installer-amd64/"
VM_OS="ubuntu24.04"
VM_IMG="/var/lib/libvirt/images/${VM_NAME}.qcow1"
VM_CORES=12
VM_DISKSIZE=10
VM_RAMSIZE=128000
VM_NET="default"


sudo virt-install \
--name ${VM_NAME} \
--memory ${VM_RAMSIZE} \
--vcpus ${VM_CORES} \
--os-variant=${VM_OS} \
--virt-type=kvm \
--network network=${VM_NET},model=virtio \
--graphics vnc \
--disk path=${VM_IMG},size=${VM_DISKSIZE},bus=virtio,format=qcow2 \
--disk path="/var/lib/libvirt/images/seed.iso",device=cdrom \
--cdrom ${VM_ISO} \
