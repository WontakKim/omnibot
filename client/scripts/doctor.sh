#!/bin/bash

# docker
if ! command -v docker &> /dev/null; then
  echo "🔴docker is not installed"
  exit 1
fi
echo "🟢$(docker --version)"

# cpu virtualization
if ! grep -E "(vmx|svm)" /proc/cpuinfo &> /dev/null; then
  echo "🔴cpu virtualization not supported (enable in BIOS)"
  exit 1
fi
echo "🟢cpu virtualization supported"

# kvm
if [ ! -e /dev/kvm ]; then
  echo "🔴/dev/kvm not found"
  exit 1
fi
if [ ! -r /dev/kvm ] || [ ! -w /dev/kvm ]; then
  echo "🔴/dev/kvm no permission"
  exit 1
fi
echo "🟢/dev/kvm ok"
