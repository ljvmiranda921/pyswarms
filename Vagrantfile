# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|

config.vm.box = "ubuntu/xenial64"

config.vm.box_check_update = false

config.vm.provision :shell, :privileged => false, :path => "provisioner.sh"
end