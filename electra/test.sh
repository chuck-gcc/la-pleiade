# #!/bin/bash
# set -e  # Stoppe le script si une commande échoue

# # === Configuration de la VM ===
# VM_NAME="test"
# VM_ISO="/var/lib/libvirt/images/ubuntu-24.04.3-desktop-amd64.iso"
# VM_SEED="/var/lib/libvirt/images/seed.iso"
# VM_IMG="/var/lib/libvirt/images/${VM_NAME}.qcow2"

# VM_OS="ubuntu24.04"
# VM_NET="default"

# VM_CORES=12
# VM_DISKSIZE=10  # en Go
# VM_RAMSIZE=128000  # en Mo (≈128 Go)

# # === Création de la VM ===
# sudo virt-install \
#   --name "${VM_NAME}" \
#   --ram "${VM_RAMSIZE}" \
#   --vcpus "${VM_CORES}" \
#   --os-variant "${VM_OS}" \
#   --virt-type kvm \
#   --network network="${VM_NET}",model=virtio \
#   --graphics vnc,listen=0.0.0.0 \
#   --disk path="${VM_IMG}",size="${VM_DISKSIZE}",bus=virtio,format=qcow2 \
#   --disk path="${VM_SEED}",device=cdrom \
#   --location "${VM_ISO}",kernel=casper/vmlinuz,initrd=casper/initrd \
#   --extra-args "autoinstall ds=nocloud-net;s=file:///cdrom/ console=ttyS0" \
#   --boot uefi \


